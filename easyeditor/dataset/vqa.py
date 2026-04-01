import os
import typing

import torch
import transformers
from PIL import Image

from .processor.base_dataset import BaseDataset
from .processor.blip_processors import BlipImageEvalProcessor
from ..trainer.utils import dict_to


class FGVEditDataset(BaseDataset):
    def __init__(
        self,
        data_dir: str,
        size: typing.Optional[int] = None,
        config=None,
        *args,
        **kwargs,
    ):
        if config is None:
            raise ValueError("FGVEditDataset requires a config.")

        self.classifier_tok = None
        if hasattr(config, "cls_processor_class"):
            self.classifier_tok = getattr(transformers, config.cls_processor_class).from_pretrained(
                config.cls_name,
                cache_dir="./hugging_cache",
            )

        model_class = getattr(config, "model_class", None)
        if config.model_name not in {"blip2", "minigpt4"} and model_class != "Blip2OPT":
            raise NotImplementedError(f"Unsupported model configuration: {config.model_name}")

        vis_processor = BlipImageEvalProcessor(image_size=364, mean=None, std=None)
        tok_name = config.tokenizer_name if config.tokenizer_name is not None else config.name
        tokenizer = getattr(transformers, config.tokenizer_class).from_pretrained(
            tok_name,
            trust_remote_code=True,
            cache_dir="./hugging_cache",
        )
        if tokenizer.pad_token in (None, ""):
            tokenizer.pad_token = tokenizer.eos_token

        super().__init__(vis_processor, config.coco_image, config.rephrase_image, [data_dir])

        self.config = config
        self.tok = tokenizer
        self.max_length = 32
        self.prompt = "Question: {} Short answer:"

        if size is not None:
            self.annotation = self.annotation[:size]

        data = []
        for record in self.annotation:
            if record["alt"] == "":
                continue

            image_path = os.path.join(self.vis_root, record["image"])
            image = self.vis_processor(Image.open(image_path).convert("RGB"))

            data.append(
                {
                    "prompt": record["src"],
                    "target": record["alt"],
                    "rephrase_prompt": record["rephrase"],
                    "image": image,
                    "image_path": image_path,
                    "image_rephrase": image,
                    "image_rephrase_path": image_path,
                    "cond": f" >> {record['alt']} || {record['src']}",
                    "fg_gen_q": record["fg_gen_q"],
                    "fg_gen_a": record["fg_gen_a"],
                    "locality_prompt": record["loc"],
                    "locality_ground_truth": record["loc_ans"],
                    "fg_loc_image": image,
                    "fg_loc_image_path": image_path,
                    "fg_loc_q": record["fg_loc_q"],
                    "fg_loc_a": record["fg_loc_a"],
                }
            )

        self._data = data

    def __getitem__(self, index):
        return self._data[index]

    def __len__(self):
        return len(self._data)

    def collate_fn(self, batch):
        src = [b["prompt"] for b in batch]
        trg = [" " + b["target"] for b in batch]
        cond = [b["cond"] for b in batch]
        rephrase = [b["rephrase_prompt"] for b in batch]
        image = [b["image"] for b in batch]
        loc_q = [b["locality_prompt"] for b in batch]
        loc_a = [" " + b["locality_ground_truth"] for b in batch]
        fg_loc_image = [b["fg_loc_image"] for b in batch]
        fg_loc_q = [b["fg_loc_q"] for b in batch]
        fg_loc_a = [" " + b["fg_loc_a"] for b in batch]
        fg_gen_image = [b["image_rephrase"] for b in batch]
        fg_gen_q = [b["fg_gen_q"] for b in batch]
        fg_gen_a = [" " + b["fg_gen_a"] for b in batch]

        if self.classifier_tok is not None and "siglip" in self.config.cls_name:
            image_clip = self.classifier_tok(
                images=[Image.open(b["image_path"]).convert("RGB") for b in batch],
                padding="max_length",
                return_tensors="pt",
            )["pixel_values"]
            fg_loc_image_clip = self.classifier_tok(
                images=[Image.open(b["fg_loc_image_path"]).convert("RGB") for b in batch],
                padding="max_length",
                return_tensors="pt",
            )["pixel_values"]
            fg_gen_image_clip = self.classifier_tok(
                images=[Image.open(b["image_rephrase_path"]).convert("RGB") for b in batch],
                padding="max_length",
                return_tensors="pt",
            )["pixel_values"]
        elif self.classifier_tok is not None:
            image_clip = self.classifier_tok(
                images=[Image.open(b["image_path"]).convert("RGB") for b in batch],
                return_tensors="pt",
            )["pixel_values"]
            fg_loc_image_clip = self.classifier_tok(
                images=[Image.open(b["fg_loc_image_path"]).convert("RGB") for b in batch],
                return_tensors="pt",
            )["pixel_values"]
            fg_gen_image_clip = self.classifier_tok(
                images=[Image.open(b["image_rephrase_path"]).convert("RGB") for b in batch],
                return_tensors="pt",
            )["pixel_values"]
        else:
            image_clip = None
            fg_loc_image_clip = None
            fg_gen_image_clip = None

        def encode_targets(prompts, labels):
            prompt_lens = [
                len(self.tok.encode(prompt, add_special_tokens=False)) for prompt in prompts
            ]
            label_ids = self.tok(
                labels,
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"]
            return prompt_lens, label_ids

        edit_inner = {
            "image": torch.stack(image, dim=0),
            "image_clip": image_clip,
            "text_input": [self.prompt.format(s) + t for s, t in zip(src, trg)],
            "text_inputt": [self.prompt.format(s) for s in src],
            "labels": trg,
        }
        edit_inner["prompts_len"], edit_inner["labels"] = encode_targets(
            [self.prompt.format(s) for s in src],
            trg,
        )

        edit_outer = {
            "image": torch.stack(image, dim=0),
            "image_clip": image_clip,
            "text_input": [self.prompt.format(r) + t for r, t in zip(rephrase, trg)],
            "text_inputt": [self.prompt.format(r) for r in rephrase],
            "labels": trg,
        }
        edit_outer["prompts_len"], edit_outer["labels"] = encode_targets(
            [self.prompt.format(r) for r in rephrase],
            trg,
        )

        fg_gen = {
            "image": torch.stack(fg_gen_image, dim=0),
            "image_clip": fg_gen_image_clip,
            "text_input": [self.prompt.format(q) + a for q, a in zip(fg_gen_q, fg_gen_a)],
            "text_inputt": [self.prompt.format(q) for q in fg_gen_q],
            "labels": fg_gen_a,
        }
        fg_gen["prompts_len"], fg_gen["labels"] = encode_targets(
            [self.prompt.format(q) for q in fg_gen_q],
            fg_gen_a,
        )

        loc = {
            "image": None,
            "image_clip": None,
            "text_input": [q + a for q, a in zip(loc_q, loc_a)],
            "text_inputt": loc_q,
            "labels": loc_a,
        }
        loc["prompts_len"], loc["labels"] = encode_targets(loc_q, loc_a)

        fg_loc = {
            "image": torch.stack(fg_loc_image, dim=0),
            "image_clip": fg_loc_image_clip,
            "text_input": [self.prompt.format(q) + a for q, a in zip(fg_loc_q, fg_loc_a)],
            "text_inputt": [self.prompt.format(q) for q in fg_loc_q],
            "labels": fg_loc_a,
        }
        fg_loc["prompts_len"], fg_loc["labels"] = encode_targets(
            [self.prompt.format(q) for q in fg_loc_q],
            fg_loc_a,
        )

        cond = self.tok(
            cond,
            return_tensors="pt",
            padding=True,
            max_length=self.max_length,
            truncation=True,
        ).to(self.config.device)

        batch = {
            "edit_inner": edit_inner,
            "edit_outer": edit_outer,
            "fg_gen": fg_gen,
            "loc": loc,
            "fg_loc": fg_loc,
            "cond": cond,
        }
        return dict_to(batch, self.config.device)
