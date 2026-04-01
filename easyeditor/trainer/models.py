import logging

import torch
import torch.nn as nn
import transformers
from transformers import GPT2Tokenizer, GPT2TokenizerFast

LOG = logging.getLogger(__name__)


class CastModule(nn.Module):
    def __init__(self, module: nn.Module, in_cast: torch.dtype = torch.float32, out_cast: torch.dtype = None):
        super().__init__()
        self.underlying = module
        self.in_cast = in_cast
        self.out_cast = out_cast

    def cast(self, obj, dtype):
        if dtype is None:
            return obj
        return obj.to(dtype) if isinstance(obj, torch.Tensor) else obj

    def forward(self, *args, **kwargs):
        args = tuple(self.cast(a, self.in_cast) for a in args)
        kwargs = {k: self.cast(v, self.in_cast) for k, v in kwargs.items()}
        outputs = self.underlying(*args, **kwargs)
        if isinstance(outputs, torch.Tensor):
            return self.cast(outputs, self.out_cast)
        if isinstance(outputs, tuple):
            return tuple(self.cast(o, self.out_cast) for o in outputs)
        raise RuntimeError(f"Not sure how to cast type {type(outputs)}")


class BertClassifier(torch.nn.Module):
    def __init__(self, model_name, hidden_dim=768):
        super().__init__()
        self.model = transformers.BertModel.from_pretrained(model_name, cache_dir="./hugging_cache")
        self.classifier = torch.nn.Linear(hidden_dim, 1)

    @property
    def config(self):
        return self.model.config

    def forward(self, *args, **kwargs):
        filtered_kwargs = {k: v for k, v in kwargs.items() if k != "labels"}
        return self.classifier(self.model(*args, **filtered_kwargs)[1])


def get_model(config):
    if config.model_class == "BertClassifier":
        model = BertClassifier(config.model_name)
    elif config.model_name == "blip2":
        from .blip2_models.blip2_opt import Blip2OPT

        model = Blip2OPT(
            vit_model="eva_clip_g",
            img_size=364,
            use_grad_checkpoint=True,
            vit_precision="fp32",
            freeze_vit=True,
            freeze_qformer=config.freeze_qformer,
            opt_model=config.name,
            state_dict_file=config.state_dict_file,
            qformer_name_or_path=config.qformer_name_or_path,
            qformer_checkpoint=config.qformer_checkpoint,
        )
    elif config.model_name == "minigpt4":
        from .blip2_models.mini_gpt4 import MiniGPT4

        model = MiniGPT4(
            vit_model="eva_clip_g",
            qformer_checkpoint=config.qformer_checkpoint,
            img_size=364,
            use_grad_checkpoint=True,
            vit_precision="fp32",
            freeze_vit=True,
            freeze_qformer=config.freeze_qformer,
            llama_model=config.name,
            state_dict_file=config.state_dict_file,
            qformer_name_or_path=config.qformer_name_or_path,
            pretrained_ckpt=config.pretrained_ckpt,
        )
    else:
        raise NotImplementedError(f"Unsupported model_name: {config.model_name}")

    if config.dropout is not None:
        n_reset = 0
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.p = config.dropout
                n_reset += 1
            if hasattr(m, "dropout") and isinstance(m.dropout, float):
                m.dropout = config.dropout
                n_reset += 1
            if hasattr(m, "activation_dropout") and isinstance(m.activation_dropout, float):
                m.activation_dropout = config.dropout
                n_reset += 1
        LOG.info(f"Set {n_reset} dropout modules to p={config.dropout}")

    param_names = [n for n, _ in model.named_parameters()]
    bad_inner_params = [p for p in config.inner_params if p not in param_names]
    if bad_inner_params and config.inner_params[0] not in ["Qformer", "vision_model"]:
        raise ValueError(f"Params {bad_inner_params} do not exist in model of type {type(model)}.")

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("**************\n", f"total params: {pytorch_total_params}\n", "**************\n")
    return model


def get_tokenizer(config):
    tok_name = config.tokenizer_name if config.tokenizer_name is not None else config.name
    tokenizer = getattr(transformers, config.tokenizer_class).from_pretrained(tok_name, cache_dir="./hugging_cache")
    if isinstance(tokenizer, GPT2Tokenizer) or isinstance(tokenizer, GPT2TokenizerFast):
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
    return tokenizer
