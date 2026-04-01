import logging
import time

import torch
from torch import nn
from torch.utils.data import Dataset

from .BaseTrainer import BaseTrainer
from .losses import kl_loc_loss
from .utils import RunningStatAverager, safe_backward

LOG = logging.getLogger(__name__)


class MultimodalTrainer(BaseTrainer):
    def __init__(self, config, train_set: Dataset, val_set: Dataset):
        super().__init__(config, train_set, val_set)

        if hasattr(self.model, "edit_lrs") and not self.config.eval_only:
            self.lr_opt = self.OptimizerClass([self.model.edit_lrs], config.lr_lr)
            if self.archive is not None:
                self.lr_opt.load_state_dict(self.archive["lr_opt"])
        elif hasattr(self.model, "replacement") and hasattr(self.model.replacement, "edit_lrs") and not self.config.eval_only:
            self.lr_opt = self.OptimizerClass([self.model.replacement.edit_lrs], self.model.replacement.config.lr_lr)
            if self.archive is not None:
                self.lr_opt.load_state_dict(self.archive["lr_opt"])
        else:
            self.lr_opt = None

        self.cls_loss_fn = nn.BCELoss()

    def _forward(self, model, batch, training):
        if self.config.alg == "MSCKE":
            return model(batch, train_in=training)
        return model(batch)

    def _extract_outputs(self, outputs, batch):
        if isinstance(outputs, torch.Tensor):
            return outputs, batch["labels"], None
        return outputs.logits, outputs.labels, getattr(outputs, "cls_sims", None)

    def edit_step(self, batch, training: bool):
        self.model.train(training)
        self.original_model.train(training)

        with torch.no_grad():
            base_text_outputs = self.model(batch["loc"])
            base_fg_loc_outputs = self.model(batch["fg_loc"])
            base_text_logits = base_text_outputs if isinstance(base_text_outputs, torch.Tensor) else base_text_outputs.logits
            base_fg_loc_logits = base_fg_loc_outputs if isinstance(base_fg_loc_outputs, torch.Tensor) else base_fg_loc_outputs.logits

        start = time.time()
        edited_model, model_info = self.model.edit(batch["edit_inner"], batch["cond"])
        edit_time = time.time() - start

        with torch.set_grad_enabled(training):
            outer_outputs = self._forward(edited_model, batch["edit_outer"], training)
            inner_outputs = self._forward(edited_model, batch["edit_inner"], training)
            fg_gen_outputs = self._forward(edited_model, batch["fg_gen"], training)
            post_text_loc_outputs = self._forward(edited_model, batch["loc"], training)
            post_fg_loc_outputs = self._forward(edited_model, batch["fg_loc"], training)

            outer_logits, outer_labels, outer_cls = self._extract_outputs(outer_outputs, batch["edit_outer"])
            inner_logits, inner_labels, inner_cls = self._extract_outputs(inner_outputs, batch["edit_inner"])
            fg_gen_logits, fg_gen_labels, fg_gen_cls = self._extract_outputs(fg_gen_outputs, batch["fg_gen"])
            post_text_loc_logits, _, text_loc_cls = self._extract_outputs(post_text_loc_outputs, batch["loc"])
            post_fg_loc_logits, _, fg_loc_cls = self._extract_outputs(post_fg_loc_outputs, batch["fg_loc"])

            text_loc_mask = (
                post_text_loc_outputs.attention_mask
                if not isinstance(post_text_loc_outputs, torch.Tensor)
                else torch.ones(post_text_loc_logits.shape[:2], device=post_text_loc_logits.device)
            )
            fg_loc_mask = (
                post_fg_loc_outputs.attention_mask
                if not isinstance(post_fg_loc_outputs, torch.Tensor)
                else torch.ones(post_fg_loc_logits.shape[:2], device=post_fg_loc_logits.device)
            )

            l_edit = self.model.edit_loss_fn(self.config, outer_logits, outer_labels, multimodal=True)["nll"]
            l_fg_gen = self.model.edit_loss_fn(self.config, fg_gen_logits, fg_gen_labels, multimodal=True)["nll"]
            l_text_loc = kl_loc_loss(base_text_logits.detach(), post_text_loc_logits, mask=text_loc_mask)
            l_fg_loc = kl_loc_loss(base_fg_loc_logits.detach(), post_fg_loc_logits, mask=fg_loc_mask)

            with torch.no_grad():
                outer_metrics = self.model.edit_loss_fn(self.config, outer_logits, outer_labels, multimodal=True)
                inner_metrics = self.model.edit_loss_fn(self.config, inner_logits, inner_labels, multimodal=True)
                fg_gen_metrics = self.model.edit_loss_fn(self.config, fg_gen_logits, fg_gen_labels, multimodal=True)

        if self.config.alg == "MSCKE":
            cls_loss = (
                self.cls_loss_fn(outer_cls, torch.ones_like(outer_cls))
                + self.config.igen_loss * self.cls_loss_fn(fg_gen_cls, torch.ones_like(fg_gen_cls))
                + self.cls_loss_fn(inner_cls, torch.ones_like(inner_cls))
                + self.cls_loss_fn(text_loc_cls, torch.zeros_like(text_loc_cls))
                + self.config.iloc_loss * self.cls_loss_fn(fg_loc_cls, torch.zeros_like(fg_loc_cls))
            )
            loc_loss = self.config.cloc * (l_text_loc + l_fg_loc)
            l_total_edit = self.config.cedit * l_edit + self.config.iedit * l_fg_gen + self.config.edit_loss * loc_loss + self.config.cls_loss * cls_loss
        else:
            l_total_edit = self.config.cedit * l_edit + self.config.cloc * (l_text_loc + l_fg_loc) + self.config.iedit * l_fg_gen

        if training and self.config.alg != "ft":
            safe_backward(l_total_edit, self.model.outer_parameters(), self.config.accumulate_bs, allow_unused=True)

        text_loc_topk = torch.topk(torch.nn.functional.softmax(post_text_loc_logits, dim=-1), k=1, dim=-1).indices
        base_text_topk = torch.topk(torch.nn.functional.softmax(base_text_logits, dim=-1), k=1, dim=-1).indices
        fg_loc_topk = torch.topk(torch.nn.functional.softmax(post_fg_loc_logits, dim=-1), k=10, dim=-1).indices
        base_fg_loc_topk = torch.topk(torch.nn.functional.softmax(base_fg_loc_logits, dim=-1), k=10, dim=-1).indices

        info_dict = {
            "loss/edit": l_edit.item(),
            "loss/fg_gen": l_fg_gen.item(),
            "loss/loc": l_text_loc.item(),
            "loss/fg_loc": l_fg_loc.item(),
            "edit/acc": outer_metrics["acc"].item(),
            "edit/log_prob": outer_metrics["log_prob"].item(),
            "edit/prob": outer_metrics["prob"].item(),
            "inner/acc": inner_metrics["acc"].item(),
            "reliability/acc": inner_metrics["acc"].item(),
            "generality/acc": outer_metrics["acc"].item(),
            "fg_gen/acc": fg_gen_metrics["acc"].item(),
            "time/edit": edit_time,
            "loc/acc": (text_loc_topk.view(-1) == base_text_topk.view(-1)).float().mean().item(),
            "fg_loc/acc": (fg_loc_topk.view(-1) == base_fg_loc_topk.view(-1)).float().mean().item(),
        }
        info_dict["specificity/acc"] = (info_dict["fg_loc/acc"] + info_dict["fg_gen/acc"]) / 2

        if self.config.alg == "MSCKE":
            info_dict["edit/cls"] = (outer_cls >= 0.5).to(torch.int).item()
            info_dict["fg_gen/cls"] = (fg_gen_cls >= 0.5).to(torch.int).item()
            info_dict["inner/cls"] = (inner_cls >= 0.5).to(torch.int).item()
            info_dict["loc/cls"] = (text_loc_cls < 0.5).to(torch.int).item()
            info_dict["fg_loc/cls"] = (fg_loc_cls < 0.5).to(torch.int).item()

        l_base = torch.tensor(0.0)
        l_total = l_total_edit + self.config.cbase * l_base
        info_dict["loss/total"] = l_total.item()
        info_dict["loss/total_edit"] = l_total_edit.item()
        info_dict["memory/alloc_max"] = torch.cuda.max_memory_allocated()
        info_dict["memory/res_max"] = torch.cuda.max_memory_reserved()
        info_dict = {**info_dict, **model_info}

        return l_total, l_edit, l_text_loc, l_base, info_dict

    def train_step(self, batch):
        _, _, _, _, info_dict = self.edit_step(batch, training=True)

        if self.global_iter > 0 and self.global_iter % self.config.accumulate_bs == 0:
            grad = torch.nn.utils.clip_grad_norm_(
                self.model.outer_parameters(),
                self.config.grad_clip,
                error_if_nonfinite=True,
            )
            info_dict["grad"] = grad.item()
            self.opt.step()
            self.opt.zero_grad()

            if self.lr_opt is not None:
                self.lr_opt.step()
                self.lr_opt.zero_grad()

                if hasattr(self.model, "replacement") and hasattr(self.model.replacement, "edit_lrs"):
                    for lr_idx, lr in enumerate(self.model.replacement.edit_lrs):
                        info_dict[f"lr/lr{lr_idx}"] = lr.item()
                elif hasattr(self.model, "edit_lrs"):
                    for lr_idx, lr in enumerate(self.model.edit_lrs):
                        info_dict[f"lr/lr{lr_idx}"] = lr.item()

        return info_dict

    def _inline_validation_log(self, step, stats, start_time, steps):
        elapsed = (time.time() - start_time) / (step + 1)
        prog = f"{step+1}/{steps}".ljust(20)
        reliability = f"{stats['reliability/acc_val']:<12.5f}"
        generality = f"{stats['generality/acc_val']:<12.5f}"
        specificity = f"{stats['specificity/acc_val']:<12.5f}"
        fg_gen = f"{stats['fg_gen/acc_val']:<12.5f}"
        fg_loc = f"{stats['fg_loc/acc_val']:<12.5f}"
        LOG.info(
            f"Step {prog} reliability: {reliability} generality: {generality} specificity: {specificity} fg_gen: {fg_gen} fg_loc: {fg_loc} it_time: {elapsed:.4f}"
        )

    def validate(self, steps=None, log: bool = False):
        if steps is None or steps > len(self.val_set):
            steps = len(self.val_set)

        if log:
            LOG.info(f"Beginning evaluation for {steps} steps...")
        averager = RunningStatAverager("val")

        start_time = time.time()
        for val_step, batch in enumerate(self.val_loader):
            if val_step >= steps:
                break
            _, _, _, _, info_dict = self.edit_step(batch, training=False)
            averager.add(info_dict)

            if log and (val_step + 1) % self.config.log_interval == 0:
                self._inline_validation_log(val_step, averager.average(), start_time, steps)

        if log:
            self._inline_validation_log(val_step, averager.average(), start_time, steps)
        elapsed = time.time() - start_time
        stats = averager.average()
        stats["eval_time/elapsed"] = elapsed
        stats["eval_time/average"] = elapsed / steps
        return stats
