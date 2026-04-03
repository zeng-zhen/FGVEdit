import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import transformers
import logging
from ..utils import scr, set_dropout, _logits, add_padding, add_sep
from .editable_model import EditableModel
from ..models import BertClassifier
from transformers import GPT2Tokenizer, GPT2TokenizerFast

LOG = logging.getLogger(__name__)

class CrossAttention(nn.Module):
    def __init__(self, text_dim, image_dim, output_dim):
        super(CrossAttention, self).__init__()
        self.text_projection = nn.Linear(text_dim, output_dim)
        self.image_projection = nn.Linear(image_dim, output_dim)
        self.output_projection = nn.Linear(output_dim * 2, output_dim)

    def forward(self, text_features, image_features):
        text_proj = self.text_projection(text_features)
        image_proj = self.image_projection(image_features)
        attention_scores = torch.matmul(text_proj, image_proj.transpose(-1, -2))
        attention_weights = F.softmax(attention_scores, dim=-1)
        image_context = torch.matmul(attention_weights, image_proj)
        combined_features = torch.cat((text_proj, image_context), dim=-1)
        fused_output = self.output_projection(combined_features)
        return fused_output
    
def attention_fusion(text_features,image_features):
    attention_scores = torch.matmul(text_features, image_features.transpose(-1, -2))
    attention_weights = F.softmax(attention_scores, dim=-1)
    fused_features = torch.matmul(attention_weights, image_features)
    combined_features = torch.cat((text_features, fused_features), dim=-1)
    return combined_features

def cat_fusion(text_features,image_features):
    combined_features = torch.cat((text_features, image_features), dim=-1)

    return combined_features

def translate_tokens(tokens, from_tok, to_tok):
    tokens = tokens.masked_fill(tokens == -100, from_tok.pad_token_id)
    text = from_tok.batch_decode(tokens, skip_special_tokens=True)
    return to_tok(text, return_tensors="pt")["input_ids"].to(tokens.device)

class MSCKE(EditableModel):
    def __init__(self, model, config, model_constructor, classifier=None, classifier_tok=None,
                 replacement=None, replacement_tok=None, cache_inputs=None, cache_labels=None,
                 cache_images=None,cache_images_clip=None,scale=None):
        super().__init__(model, config, model_constructor)
        if classifier is None:
            self.classifier = getattr(transformers, config.cls_class).from_pretrained(config.cls_name, cache_dir='./hugging_cache')
            if self.config.checkpoint_grad:
                LOG.info(f"Checking for checkpointing: {hasattr(self.classifier.config, 'gradient_checkpointing')}")
                self.classifier.config.gradient_checkpointing = True
            self.classifier_tok = getattr(transformers, config.cls_processor_class).from_pretrained(config.cls_name, cache_dir='./hugging_cache')
            set_dropout(self.classifier, config.dropout)
        else:
            assert isinstance(classifier, torch.nn.Module), f"Classifier is a {type(classifier)}!"
            assert isinstance(classifier_tok, transformers.ProcessorMixin), f"Classifier tok is {type(classifier_tok)}!"
            self.classifier, self.classifier_tok = classifier, classifier_tok

        if replacement is None:
            if config.model_name == "minigpt4":
                self.replacement_tok = transformers.LlamaTokenizer.from_pretrained(config.small_name, cache_dir='./hugging_cache')
                self.replacement_tok.pad_token = self.replacement_tok.eos_token
            else:
                self.replacement_tok = transformers.AutoTokenizer.from_pretrained(config.small_name, cache_dir='./hugging_cache')
            if self.config.freeze_cntr:
                self.replacement = None
            else:
                if config.model_class == "BertClassifier":
                    self.replacement = BertClassifier(config.small_name)
                elif config.model_name == "blip2":
                    if "opt" in config.name:
                        from transformers import OPTForCausalLM
                        self.replacement = OPTForCausalLM.from_pretrained(config.small_name, cache_dir='./hugging_cache')
                        if "2.7b" in config.small_name:
                            for k, v in self.replacement.named_parameters():
                                if '31' in k:
                                    v.requires_grad = True
                                else:
                                    v.requires_grad = False
                elif config.model_name == "minigpt4":
                    from transformers import LlamaForCausalLM
                    self.replacement = LlamaForCausalLM.from_pretrained(config.small_name, cache_dir='./hugging_cache')
                    for k, v in self.replacement.named_parameters():
                        if '31' in k:
                            v.requires_grad = True
                        else:
                            v.requires_grad = False
                else:
                    self.replacement = getattr(transformers, config.model_class).from_pretrained(config.small_name, cache_dir='./hugging_cache')
                if self.replacement_tok.sep_token is None and "gpt" not in config.name.lower():
                    add_sep(self.replacement_tok, self.replacement)
                if self.replacement_tok.pad_token is None:
                    add_padding(self.replacement_tok, self.replacement)
                set_dropout(self.replacement, config.dropout)
        else:
            assert isinstance(replacement, torch.nn.Module), f"Rep is {type(replacement)}!"
            assert isinstance(replacement_tok, transformers.PreTrainedTokenizerBase), f"Rep tok is {type(replacement_tok)}!"
            self.replacement, self.replacement_tok = replacement, replacement_tok
        if self.config.cross_attend:
            self.scale = None
        else:
            if scale is None:
                self.register_buffer("scale", torch.tensor(1.0))
            else:
                self.scale = scale
        self.language_projection = torch.nn.Linear(self.model.Qformer.config.hidden_size, self.replacement.config.hidden_size)
        if self.config.fusion_type == "cross":
            self.cross_attend = CrossAttention(self.classifier.text_model.config.projection_dim,self.classifier.vision_model.config.projection_dim,512)
        if cache_inputs is None:
            self.cache_inputs = []
            self.cache_labels = []
            self.cache_images = []
            self.cache_images_clip = []
        else:
            assert isinstance(cache_inputs, list), f"Cache inputs is {cache_inputs}"
            assert isinstance(cache_labels, list), f"Cache labels is {cache_labels}"
            assert isinstance(cache_images, list), f"Cache labels is {cache_images}"
            assert isinstance(cache_images_clip, list), f"Cache labels is {cache_images_clip}"
            self.cache_inputs = copy.deepcopy(cache_inputs)
            self.cache_labels = copy.deepcopy(cache_labels)
            self.cache_images = copy.deepcopy(cache_images)
            self.cache_images_clip = copy.deepcopy(cache_images_clip)

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        state_dict = super().state_dict(prefix=prefix, keep_vars=keep_vars)  # Get default state dict
        model_keys = self.model.state_dict(prefix=prefix, keep_vars=keep_vars).keys()  # Remove model params
        for k in model_keys:
            del state_dict[f"model.{k}"]
        if self.config.freeze_cntr:
            cntr_keys = self.replacement.state_dict().keys()
            for k in cntr_keys:
                del state_dict[f"replacement.{k}"]
        state_dict["model_config"] = self.model.config  # Include model config
        return state_dict

    def load_state_dict(self, state_dict, strict: bool = True):
        config = state_dict["model_config"]
        del state_dict["model_config"]
        if config != self.model.config:
            LOG.info("Loaded model config doesn't match current model config.")
            LOG.info(f"Loaded: {config}")
            LOG.info(f"Current: {self.model.config}")

        if self.config.freeze_cntr:
            rep_keys = list(state_dict.keys())
            for k in rep_keys:
                if k.startswith("replacement"):
                    del state_dict[k]
            res = super().load_state_dict(state_dict, False)
        else:
            res = super().load_state_dict(state_dict, False)

        # We should only have missing keys for the model, and no unexpected keys
        def ok_to_miss(k):
            return k.startswith("model.") or (self.config.freeze_cntr and k.startswith("replacement."))
        missing_keys = [k for k in res.missing_keys if not ok_to_miss(k)]
        assert len(missing_keys) == 0, f"Should only have missing keys for model: {missing_keys}."
        assert len(res.unexpected_keys) == 0, "Shouldn't have any unexpected keys"
        return res

    def outer_parameters(self, grouped=False):
        if self.config.freeze is not None:
            modlist = None
            for m in self.classifier.modules():
                if isinstance(m, torch.nn.ModuleList):
                    modlist = m
                    break
            model_params = list(modlist[-self.config.freeze:].parameters())
        else:
            model_params = list(self.classifier.parameters())

        if self.config.freeze is not None:
            cls = self.classifier
            if hasattr(cls, "classifier"):
                model_params.extend(cls.classifier.parameters())
            if hasattr(cls, "pre_classifier"):
                model_params.extend(cls.pre_classifier.parameters())

        if not self.config.freeze_cntr:
            if self.config.model_name == "minigpt4" or "opt-2.7b" in self.config.small_name:
                params_extend = []
                # alter
                for k, v in self.replacement.named_parameters():
                    if '31' in k:
                        params_extend.append(v)
                model_params.extend(params_extend)
            else:
                model_params.extend(list(self.replacement.parameters()))

        extra_params = []
        if grouped:
            return [
                dict(params=model_params, lr=self.config.lr),
                dict(params=extra_params, lr=self.config.lr_lr)
            ]
        else:
            return model_params + extra_params

    def edit(self, batch, condition=None, detach_history=False):
        def detokenize(toks, tok):
            tokens = toks.masked_fill(toks == -100, tok.pad_token_id)
            return tok.batch_decode(tokens, skip_special_tokens=True)
        if self.config.model_name == "minigpt4" or self.config.model_name == "blip2":
            inputs = batch["text_input"] # src, trg
        else:
            inputs = detokenize(batch["input_ids"], self.replacement_tok)
        if "bert" in self.config.name:
            labels = ["" for _ in batch["labels"]]
        elif self.config.model_name == "minigpt4" or self.config.model_name == "blip2":
            labels = batch["labels"]
            if isinstance(labels, torch.Tensor):
                labels = detokenize(labels, self.replacement_tok)
        else:
            labels = detokenize(batch["labels"], self.replacement_tok)
        
        if batch['image'] is None:
            images = torch.zeros((3, 364, 364))*len(batch["text_input"])
            # images_clip = torch.zeros((3, 384, 384))*len(batch["text_input"])
            images_clip = torch.zeros((3, 224, 224))*len(batch["text_input"])
        else:
            images = [batch['image'][i] for i in range(batch['image'].size(0))]
            images_clip = [batch['image_clip'][i] for i in range(batch['image_clip'].size(0))]

        cache_inputs = self.cache_inputs + inputs
        cache_labels = self.cache_labels + labels
        cache_images = self.cache_images + images
        cache_images_clip = self.cache_images_clip + images_clip

        new_model = MSCKE(self.model, self.config, self.model_constructor, self.classifier, self.classifier_tok,
                        self.replacement, self.replacement_tok, cache_inputs, cache_labels, cache_images, cache_images_clip, self.scale)
        new_model.train(self.training)
        return new_model, {}

    def stats(self):
        return self.last_stats


    # def multi_embedding_logsim_matrix(self, ctx_text, input_text, ctx_images=None, input_images=None):
    #     ctx_text_inputs = self.classifier_tok(text=ctx_text, return_tensors="pt", padding="max_length").to(self.config.device)
    #     ctx_images_inputs = torch.stack(ctx_images)
    #     if input_images is None:
    #         input_images = torch.zeros((len(input_text),3, 384, 384)).to(self.config.device)
    #     main_text_inputs = self.classifier_tok(text=input_text, return_tensors="pt", padding="max_length").to(self.config.device)
    #     main_images_inputs = input_images
    #     if self.config.emb_type == "emb":
    #         ctx_inputs = {
    #             "input_ids": ctx_text_inputs["input_ids"],
    #             "attention_mask": ctx_text_inputs["attention_mask"],
    #             "pixel_values": ctx_images_inputs
    #         }
    #         ctx_emb = self.classifier(**ctx_inputs)
    #         ctx_text_emb = ctx_emb.text_embeds
    #         ctx_image_emb = ctx_emb.image_embeds
    #         main_inputs = {
    #             "input_ids": main_text_inputs["input_ids"],
    #             "attention_mask": main_text_inputs["attention_mask"],
    #             "pixel_values": main_images_inputs
    #         }
    #         main_emb = self.classifier(**main_inputs)
    #         main_emb = self.classifier(**ctx_inputs)
    #         main_text_emb = main_emb.text_embeds
    #         main_image_emb = main_emb.image_embeds
    #     elif self.config.emb_type == "get_features":
    #         ctx_inputs = {
    #             "input_ids": ctx_text_inputs["input_ids"],
    #             #  "attention_mask": ctx_text_inputs["attention_mask"]
    #         }
    #         ctx_text_emb = self.classifier.get_text_features(**ctx_inputs)
    #         ctx_image_emb = self.classifier.get_image_features(ctx_images_inputs)
    #         main_inputs = {
    #             "input_ids": main_text_inputs["input_ids"],
    #             #  "attention_mask": main_text_inputs["attention_mask"]
    #         }
    #         main_text_emb = self.classifier.get_text_features(**main_inputs)
    #         main_image_emb = self.classifier.get_image_features(main_images_inputs)
        
    #     if self.config.fusion_type == "cross":
    #         self.cross_attend = self.cross_attend.to(ctx_text_emb.device)
    #         ctx_fused_embdes = self.cross_attend(ctx_text_emb, ctx_image_emb).unsqueeze(1)
    #         main_fused_embdes = self.cross_attend(main_text_emb, main_image_emb).unsqueeze(1)
    #     elif self.config.fusion_type == "dot":
    #         ctx_fused_embdes = attention_fusion(ctx_text_emb, ctx_image_emb).unsqueeze(1)
    #         main_fused_embdes = attention_fusion(main_text_emb, main_image_emb).unsqueeze(1)
    #     elif self.config.fusion_type == "text":
    #         ctx_fused_embdes = ctx_text_emb.unsqueeze(1)
    #         main_fused_embdes = main_text_emb.unsqueeze(1)
    #     elif self.config.fusion_type == "cat":
    #         ctx_fused_embdes = cat_fusion(ctx_text_emb, ctx_image_emb).unsqueeze(1)
    #         main_fused_embdes = cat_fusion(main_text_emb, main_image_emb).unsqueeze(1)

    #     ctx_fused_embdes = ctx_fused_embdes.view(ctx_fused_embdes.shape[0], self.config.dist_heads, -1)
    #     main_fused_embdes = main_fused_embdes.view(main_fused_embdes.shape[0], self.config.dist_heads, -1)

    def multi_embedding_logsim_matrix(self, ctx_text, input_text, ctx_images=None, input_images=None):
        ctx_text_inputs = self.classifier_tok(text=ctx_text, return_tensors="pt", padding=True).to(self.config.device)
        ctx_images_inputs = torch.stack(ctx_images)
        if input_images is None:
            input_images = torch.zeros((len(input_text),3, 224, 224)).to(self.config.device)
        main_text_inputs = self.classifier_tok(text=input_text, return_tensors="pt", padding=True).to(self.config.device)
        main_images_inputs = input_images
        if self.config.emb_type == "emb":
            ctx_inputs = {
                "input_ids": ctx_text_inputs["input_ids"],
                "attention_mask": ctx_text_inputs["attention_mask"],
                "pixel_values": ctx_images_inputs
            }
            ctx_emb = self.classifier(**ctx_inputs)
            ctx_text_emb = ctx_emb.text_embeds
            ctx_image_emb = ctx_emb.image_embeds
            main_inputs = {
                "input_ids": main_text_inputs["input_ids"],
                "attention_mask": main_text_inputs["attention_mask"],
                "pixel_values": main_images_inputs
            }
            main_emb = self.classifier(**main_inputs)
            main_emb = self.classifier(**ctx_inputs)
            main_text_emb = main_emb.text_embeds
            main_image_emb = main_emb.image_embeds
        elif self.config.emb_type == "get_features":
            ctx_inputs = {
                "input_ids": ctx_text_inputs["input_ids"],
                "attention_mask": ctx_text_inputs["attention_mask"]
            }
            ctx_text_emb = self.classifier.get_text_features(**ctx_inputs)
            ctx_image_emb = self.classifier.get_image_features(ctx_images_inputs)
            main_inputs = {
                "input_ids": main_text_inputs["input_ids"],
                "attention_mask": main_text_inputs["attention_mask"]
            }
            main_text_emb = self.classifier.get_text_features(**main_inputs)
            main_image_emb = self.classifier.get_image_features(main_images_inputs)
        
        if self.config.fusion_type == "cross":
            self.cross_attend = self.cross_attend.to(ctx_text_emb.device)
            ctx_fused_embdes = self.cross_attend(ctx_text_emb, ctx_image_emb).unsqueeze(1)
            main_fused_embdes = self.cross_attend(main_text_emb, main_image_emb).unsqueeze(1)
        elif self.config.fusion_type == "dot":
            ctx_fused_embdes = attention_fusion(ctx_text_emb, ctx_image_emb).unsqueeze(1)
            main_fused_embdes = attention_fusion(main_text_emb, main_image_emb).unsqueeze(1)
        elif self.config.fusion_type == "text":
            ctx_fused_embdes = ctx_text_emb.unsqueeze(1)
            main_fused_embdes = main_text_emb.unsqueeze(1)
        elif self.config.fusion_type == "cat":
            ctx_fused_embdes = cat_fusion(ctx_text_emb, ctx_image_emb).unsqueeze(1)
            main_fused_embdes = cat_fusion(main_text_emb, main_image_emb).unsqueeze(1)

        ctx_fused_embdes = ctx_fused_embdes.view(ctx_fused_embdes.shape[0], self.config.dist_heads, -1)
        main_fused_embdes = main_fused_embdes.view(main_fused_embdes.shape[0], self.config.dist_heads, -1)

        

        if self.config.bound_embeds:
            ctx_fused_embdes = ctx_fused_embdes.tanh()
            main_fused_embdes = main_fused_embdes.tanh()

        if self.config.cos:
            cos = (ctx_fused_embdes[None] * main_fused_embdes[:, None]).sum(-1) / (ctx_fused_embdes[None].norm(2, -1) * main_fused_embdes[:, None].norm(2, -1))
            dists = 1 - cos
        else:
            dists = (ctx_fused_embdes[None] - main_fused_embdes[:, None]).norm(2, -1)
            if self.config.square:
                dists = dists ** 2

        dists = dists.min(-1).values

        assert dists.min() >= 0, "Shouldn't have negative distances!"

        cls_logsims = -dists * self.scale

        return cls_logsims
    

    def build_rep_cache_contexts(self):
        sep = " "
        if hasattr(self.model, "name_or_path") and "gpt" in self.model.name_or_path.lower():
            # The labels are include in the inputs for autoregressive models. Cut off the label for the classifier
            ctxs = [cin + sep for cin in self.cache_inputs]
        else:
            # ctxs = [cin + sep + clab + sep for cin, clab in zip(self.cache_inputs, self.cache_labels)]
            ctxs = [cin + sep for cin in self.cache_inputs]
        return ctxs

    def build_cls_cache_inputs(self):
        inputs = self.cache_inputs
        images = self.cache_images_clip
        
        return inputs, images

    def build_rep_input_tokens(self, kwargs, idxs, generation=False):
        if "input_ids" in kwargs:
            assert len(idxs) == len(kwargs["input_ids"]), "Need one cache idx for each test input"
        cache_contexts = self.build_rep_cache_contexts()
        selected_contexts = [cache_contexts[idx.item()] for idx in idxs]
        if self.config.model_name == "minigpt4" or self.config.model_name == "blip2":
            test_inputs = kwargs["text_input"]
        else:
            test_inputs = self.replacement_tok.batch_decode(kwargs["input_ids"], skip_special_tokens=True)
        rep_texts = [ctx + inp for ctx, inp in zip(selected_contexts, test_inputs)]
        rep_input_tokens = self.replacement_tok(rep_texts, return_tensors="pt", add_special_tokens=False).to(self.config.device)

        rep_kwargs = {
            "input_ids": rep_input_tokens["input_ids"],
            "attention_mask": rep_input_tokens["attention_mask"],
        }

        if not generation:
            if 'labels' in kwargs.keys():
                rep_kwargs["labels"] = kwargs["labels"]

        if self.config.model_name == "minigpt4" or self.config.model_name == "blip2":
            # Add 'ignore' labels for the prepended cache inputs
            pre = torch.full((kwargs["labels"].shape[0], rep_kwargs["input_ids"].shape[-1] - kwargs["labels"].shape[-1]), -100,
                             device=kwargs["labels"].device)
            rep_kwargs["labels"] = torch.cat((pre, kwargs["labels"]), dim=-1)
        # if self.config.model_name == "minigpt4":
            # rep_kwargs["labels"] = self.replacement_tok(rep_kwargs["labels"], return_tensors="pt", padding=True).to(self.config.device)["input_ids"]
            # rep_kwargs["labels"] = rep_kwargs["labels"]
        return rep_kwargs

    def run_classifier(self, *inputs, **kwargs):
        cache_inputs, cache_images = self.build_cls_cache_inputs()
        if self.config.model_name == "minigpt4" or self.config.model_name == "blip2":
            test_inputs = inputs[0]["text_inputt"]
        else:
            test_inputs = self.replacement_tok.batch_decode(kwargs["input_ids"], skip_special_tokens=True)
            
        test_images = inputs[0]["image_clip"]

        # if self.config.cross_attend:
        #     log_sim_matrix = self.crossattend_logsim_matrix(cache_inputs, test_inputs)
        # else:
        #     log_sim_matrix = self.embedding_logsim_matrix(cache_inputs, test_inputs)
        log_sim_matrix = self.multi_embedding_logsim_matrix(cache_inputs, test_inputs, cache_images, test_images)
        
        sims = log_sim_matrix.exp()
        assert sims.max() <= 1, "Similarities shouldn't exceed 1!"

        cls_sims, cls_idxs = sims.max(-1)
        return cls_sims, cls_idxs, log_sim_matrix

    def generate(self, *args, **kwargs):
        input_text = self.replacement_tok.batch_decode(kwargs["input_ids"], skip_special_tokens=True)

        assert len(args) == 0, "Should only pass named arguments to generate()"
        if len(self.cache_inputs) > 0:
            cls_sims, cls_idxs, _ = self.run_classifier(*args, **kwargs)
            assert cls_sims.numel() == 1
            print(f"Cache score: {cls_sims.item()} " + ("[MISS]" if cls_sims.item() < 0.5 else "[HIT]"))
            if cls_sims.item() > 0.5:
                rep_input = self.build_rep_input_tokens(kwargs, cls_idxs, generation=True)
                kwargs["input_ids"] = rep_input["input_ids"]
                kwargs["attention_mask"] = rep_input["attention_mask"]
                rep_input_text = self.replacement_tok.decode(rep_input["input_ids"][0])
                print(f"Returning counterfactual model output for '{rep_input_text}'")
                if self.config.freeze_cntr:
                    return self.model.generate(*args, **kwargs)
                else:
                    return self.replacement.generate(*args, **kwargs)

        print(f"Returning base model output for '{input_text}'")
        return self.model.generate(*args, **kwargs)

    def forward(self, *inputs, return_logits_only=True, eps=torch.finfo(torch.float32).eps, pos_pairs=None, train_in=False, **kwargs):
        grad_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(self.training)

        # need to do soft mixing of logits if we're doing supervised training or we've specifically requested it
        # soft = self.training or (not self.config.supervised) or self.config.soft_weighting
        soft = (not self.config.supervised) or self.config.soft_weighting
        with torch.no_grad():
            if len(self.cache_inputs) == 0:
                if self.config.model_name == "blip2" or self.config.model_name == "minigpt4":
                    super_out = self.model(*inputs, **kwargs)
                else:
                    super_out = super().forward(*inputs, **kwargs).float()
                torch.set_grad_enabled(grad_enabled)
                return super_out
            else:
                if self.config.model_name == "blip2" or self.config.model_name == "minigpt4":
                    # if "prompts_len" in kwargs:
                    #     prompts_len = kwargs.pop("prompts_len")
                    base_logits = self.model(*inputs, **kwargs)
                    if not isinstance(base_logits, torch.Tensor):
                        final_labels = base_logits.labels
                        final_att_mask = base_logits.attention_mask
                        base_logits = base_logits.logits
                    base_logits = base_logits.float()
                else:
                    base_logits = super().forward(*inputs, **kwargs).float()
                if soft:
                    if base_logits.dim() == 3:
                        base_probs = base_logits.softmax(-1)
                    else:
                        base_probs = base_logits.sigmoid()
                    del base_logits

        cls_sims, cls_idxs, cls_logits = self.run_classifier(*inputs, **kwargs)
        if self.config.model_name == "minigpt4" or self.config.model_name == "blip2":
            rep_cls_inputs = self.build_rep_input_tokens(inputs[0], cls_idxs)
        else:
            rep_cls_inputs = self.build_rep_input_tokens(kwargs, cls_idxs)
        if self.config.freeze_cntr:
            rep_cls_logits = super().forward(**rep_cls_inputs)
        else:
            if self.config.model_name == "blip2":
                rep_cls_labels = rep_cls_inputs.pop("labels")
                # add vision outputs
                image = inputs[0]["image"]
                # if rep_cls_inputs["input_ids"][:, -1] != 13:
                #     eos = torch.ones([rep_cls_inputs["input_ids"].shape[0], 1],
                #                      dtype=torch.long).to(rep_cls_inputs["input_ids"].device).fill_(13)
                #     eos_attn = eos.fill_(1)
                #     rep_cls_inputs["input_ids"] = torch.cat([rep_cls_inputs["input_ids"], eos], dim=1)
                #     rep_cls_inputs["attention_mask"] = torch.cat([rep_cls_inputs["attention_mask"], eos_attn], dim=1)
                if image is not None:
                    # vision_outputs = self.model.vision_model(
                    #     pixel_values=pixel_values
                    # )
                    # image_embeds = vision_outputs[0]
                    with self.model.maybe_autocast():
                        image_embeds = self.model.ln_vision(self.model.visual_encoder(image))
                    image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
                    query_tokens = self.model.query_tokens.expand(image_embeds.shape[0], -1, -1)
                    query_output = self.model.Qformer.bert(
                        query_embeds=query_tokens,
                        encoder_hidden_states=image_embeds,
                        encoder_attention_mask=image_atts,
                        return_dict=True,
                    )
                    # query_output = query_output[0] 
                    self.language_projection = self.language_projection.to(query_output.last_hidden_state.device)
                    inputs_opt = self.language_projection(query_output.last_hidden_state)
                    atts_opt = torch.ones(
                        inputs_opt.size()[:-1], dtype=torch.long, device=image.device
                    )  
                    
                    opt_tokens = rep_cls_inputs
                    targets = rep_cls_labels

                    empty_targets = (
                        torch.ones(atts_opt.size(), dtype=torch.long).to(image.device).fill_(-100)
                    )
                    targets = torch.cat([empty_targets, targets], dim=1)

                    inputs_embeds = self.replacement.model.decoder.embed_tokens(opt_tokens["input_ids"])
                    inputs_embeds = torch.cat([inputs_opt, inputs_embeds], dim=1)
                    attention_mask = torch.cat([atts_opt, opt_tokens["attention_mask"]], dim=1)                    
                    
                    rep_cls_outputs = self.replacement(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        return_dict=True,
                        labels=targets,
                    )
                    rep_cls_logits = rep_cls_outputs.logits
                else:
                    rep_cls_outputs = self.replacement(**rep_cls_inputs)
                    rep_cls_logits = rep_cls_outputs.logits
                if soft:
                    rep_cls_logits = rep_cls_logits[:, -base_probs.shape[1]:, :]
                else:
                    rep_cls_logits = rep_cls_logits[:, -base_logits.shape[1]:, :]
            elif self.config.model_name == "minigpt4":
                rep_cls_labels = rep_cls_inputs.pop("labels")
                image = inputs[0]["image"]
                # if rep_cls_inputs["input_ids"][:, -1] != 13:
                #     eos = torch.ones([rep_cls_inputs["input_ids"].shape[0], 1],
                #                      dtype=torch.long).to(rep_cls_inputs["input_ids"].device).fill_(13)
                #     eos_attn = eos.fill_(1)
                #     rep_cls_inputs["input_ids"] = torch.cat([rep_cls_inputs["input_ids"], eos], dim=1)
                #     rep_cls_inputs["attention_mask"] = torch.cat([rep_cls_inputs["attention_mask"], eos_attn], dim=1)
                if image is not None:
                    img_embeds, atts_img = self.model.encode_img(image)
                    prompt = '###Human: <Img><ImageHere></Img> '
                    img_embeds, atts_img = self.model.prompt_wrap(img_embeds, atts_img, prompt)
                    
                    to_regress_tokens = rep_cls_inputs
                    targets = rep_cls_labels   
                    # for i, prompt_len in enumerate(inputs[0]['prompts_len']):
                    #     targets[i, :prompt_len] = -100    
                    empty_targets = (torch.ones(atts_img.shape, dtype=torch.long).to(image.device).fill_(-100))
                    targets = torch.cat([empty_targets, targets], dim=1)

                    # batch_size = img_embeds.shape[0]
                    # bos = torch.ones([batch_size, 1],
                    #                 dtype=to_regress_tokens["input_ids"].dtype,
                    #                 device=to_regress_tokens["input_ids"].device) * self.replacement_tok.bos_token_id
                    # bos_embeds = self.replacement.model.embed_tokens(bos)
                    # atts_bos = atts_img[:, :1]

                    to_regress_embeds = self.replacement.model.embed_tokens(to_regress_tokens["input_ids"])
                    inputs_embeds = torch.cat([img_embeds, to_regress_embeds], dim=1)
                    attention_mask = torch.cat([atts_img, to_regress_tokens["attention_mask"]], dim=1) 
                    # inputs_embeds = torch.cat([bos_embeds, img_embeds, to_regress_embeds], dim=1)
                    # attention_mask = torch.cat([atts_bos, atts_img, to_regress_tokens["attention_mask"]], dim=1) 
                    
                    rep_cls_outputs = self.replacement(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        return_dict=True,
                        labels=targets,
                    )
                    rep_cls_logits = rep_cls_outputs.logits
                else:
                    rep_cls_logits = _logits(self.replacement(**rep_cls_inputs))
                if soft:
                    rep_cls_logits = rep_cls_logits[:, -base_probs.shape[1]:, :]
                else:
                    rep_cls_logits = rep_cls_logits[:, -base_logits.shape[1]:, :]
            else:
                rep_cls_logits = _logits(self.replacement(**rep_cls_inputs))

        if pos_pairs is not None:
            assert (pos_pairs[:, 0] == torch.arange(pos_pairs.shape[0], device=pos_pairs.device)).all()
            gold_idxs = pos_pairs[:, 1]
            rep_gold_inputs = self.build_rep_input_tokens(kwargs, gold_idxs)
            if self.config.freeze_cntr:
                rep_gold_logits = super().forward(**rep_gold_inputs)
            else:
                rep_gold_logits = _logits(self.replacement(**rep_gold_inputs))
        else:
            rep_gold_logits = rep_cls_logits

        cls_sims = cls_sims.view(-1, 1)  # For (binary) classification, predictions are (B x 1)
        if rep_cls_logits.dim() == 3:
            cls_sims.unsqueeze_(-1)  # For generation/seq2seq, predictions are (B x S x V)

        stats = {
            'sims/mean': cls_sims.mean().item(),
            'sims/pos': (cls_sims >= 0.5).float().mean().item(),
            'sims/neg': (cls_sims < 0.5).float().mean().item(),
            'params/scale': self.scale.item() if self.scale is not None else 0.0,
        }

        # if hasattr(self.model, "name_or_path") and "gpt" in self.model.name_or_path.lower():
        #     rep_cls_logits = rep_cls_logits[:, -kwargs["labels"].shape[-1]:, :]

        if soft:
            if base_probs.size(1) != rep_cls_logits.size(1):
                rep_cls_logits = rep_cls_logits[:, -base_probs.size(1):, :]
            rep_weight = cls_sims
            if rep_cls_logits.device != base_probs.device:
                rep_cls_logits = rep_cls_logits.to(base_probs.device)
            if rep_weight.device != base_probs.device:
                rep_weight = rep_weight.to(base_probs.device)
            if base_probs.dim() == 3:
                mixture_logits = ((1 - rep_weight) * base_probs + rep_weight * rep_cls_logits.softmax(-1) + eps).log()
            else:
                mixture_logits = ((1 - rep_weight) * base_probs + rep_weight * rep_cls_logits.sigmoid() + eps).log()
        elif train_in:
            if base_logits.size(1) != rep_cls_logits.size(1):
                rep_cls_logits = rep_cls_logits[:, -base_logits.size(1):, :]
            rep_idxs = torch.where(cls_sims > 0)[0]
            mixture_logits = base_logits
            if rep_idxs.numel() > 0:
                if rep_cls_logits.device != mixture_logits.device:
                    rep_cls_logits.to(mixture_logits.device)
                mixture_logits[rep_idxs] = rep_cls_logits[rep_idxs]
        else:
            if base_logits.size(1) != rep_cls_logits.size(1):
                rep_cls_logits = rep_cls_logits[:, -base_logits.size(1):, :]
            rep_idxs = torch.where(cls_sims > 0.5)[0]
            mixture_logits = base_logits
            if rep_idxs.numel() > 0:
                if rep_cls_logits.device != mixture_logits.device:
                    rep_cls_logits.to(mixture_logits.device)
                mixture_logits[rep_idxs] = rep_cls_logits[rep_idxs]

        torch.set_grad_enabled(grad_enabled)
        if return_logits_only:
            from ..blip2_models.mini_gpt4 import MiniGPTOutput
            return MiniGPTOutput(
                    logits=mixture_logits,
                    labels=final_labels,
                    attention_mask=final_att_mask,
                    cls_sims = cls_sims,
                )
        else:
            return mixture_logits, cls_logits, rep_gold_logits, stats
