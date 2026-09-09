"""GradAudit implementation documentation."""
import os
import re
from typing import Dict, List, Optional

import torch

from gradaudit.core.victim import VictimModel
from gradaudit.core.params import select_params

# Implementation note.
KINDS = {
    "smolvlm2": {
        "remote": False,
        "text_pattern": r"model\.text_model\.layers\.(\d+)\.",
        "vision_pattern": r"model\.vision_model\.encoder\.layers\.(\d+)\.",
        "proj_keys": ("connector", "modality_projection"),
    },
    "minicpmv2": {
        "remote": True,
        "text_pattern": r"llm\.model\.layers\.(\d+)\.",
        "vision_pattern": r"vpm\.vision_model\.encoder\.layers\.(\d+)\.",
        "proj_keys": ("mm_projector",),
    },
    "deepseekvl2": {
        "remote": True,
        "text_pattern": r"language_model\.model\.layers\.(\d+)\.",
        "vision_pattern": r"vision_tower\.vision_model\.encoder\.layers\.(\d+)\.",
        "proj_keys": ("projector", "merger"),
    },
}


class GenVLMVictim(VictimModel):
    family = "mllm"

    def __init__(self, name: str, model_dir: str, kind: str,
                 device: str = "cuda:0", max_new_tokens: int = 64):
        self.name = name
        self.model_dir = model_dir
        self.kind = kind
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.model = None
        self.processor = None
        self.tokenizer = None

    # Implementation note.
    def load(self) -> None:
        kind = KINDS[self.kind]
        if self.kind == "smolvlm2":
            from transformers import AutoProcessor, SmolVLMForConditionalGeneration
            self.processor = AutoProcessor.from_pretrained(self.model_dir)
            self.model = SmolVLMForConditionalGeneration.from_pretrained(
                self.model_dir, torch_dtype=torch.bfloat16).to(self.device)
            self.tokenizer = self.processor.tokenizer
            # Implementation note.
            _orig_gif = self.model.model.get_image_features

            def _patched_gif(pixel_values, pixel_attention_mask=None):
                out = _orig_gif(pixel_values, pixel_attention_mask)
                return out.to(self.model.dtype)
            self.model.model.get_image_features = _patched_gif
        elif self.kind == "minicpmv2":
            # Implementation note.
            import torch.nn as nn
            if not hasattr(nn.Module, "_initialize_weights"):
                nn.Module._initialize_weights = lambda self, module: None
            import timm
            if not hasattr(timm.models.VisionTransformer, "_initialize_weights"):
                timm.models.VisionTransformer._initialize_weights = lambda self, module: None
            from transformers import AutoModel, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir,
                                                           trust_remote_code=True)
            self.model = AutoModel.from_pretrained(
                self.model_dir, trust_remote_code=True, torch_dtype=torch.bfloat16)
            # Implementation note.
            self.model = self.model.to(self.device)
            self.processor = self.tokenizer
            # Implementation note.
            self.processor.tokenizer = self.tokenizer
        elif self.kind == "deepseekvl2":
            from transformers import AutoModelForCausalLM, AutoProcessor
            self.processor = AutoProcessor.from_pretrained(self.model_dir,
                                                           trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_dir, trust_remote_code=True, torch_dtype=torch.bfloat16)
            self.model = self.model.to(self.device)
            self.tokenizer = self.processor.tokenizer
        else:
            raise ValueError(self.kind)
        self.model.eval()

    # Implementation note.
    def build_lm_inputs(self, image, prompt: str, target: str,
                        mask_mode: str = "assistant_only", **kw) -> Optional[dict]:
        try:
            if self.kind == "smolvlm2":
                return self._inputs_smolvlm2(image, prompt, target, mask_mode)
            if self.kind == "minicpmv2":
                return self._inputs_minicpmv2(image, prompt, target, mask_mode)
            if self.kind == "deepseekvl2":
                return self._inputs_deepseekvl2(image, prompt, target, mask_mode)
        except Exception:
            return None
        return None

    def _inputs_smolvlm2(self, image, prompt, target, mask_mode):
        msgs = [{"role": "user", "content": [{"type": "image"},
                                             {"type": "text", "text": prompt}]},
                {"role": "assistant", "content": [{"type": "text", "text": target}]}]
        full_text = self.processor.apply_chat_template(msgs, tokenize=False,
                                                       add_generation_prompt=False)
        inputs = self.processor(text=[full_text], images=[image], return_tensors="pt")
        labels = inputs["input_ids"].clone()
        if mask_mode == "assistant_only":
            user_msgs = [{"role": "user",
                          "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
            prefix_text = self.processor.apply_chat_template(
                user_msgs, tokenize=False, add_generation_prompt=True)
            prefix = self.processor(text=[prefix_text], images=[image], return_tensors="pt")
            labels[:, :prefix["input_ids"].shape[1]] = -100
        inputs["labels"] = labels
        dev = next(self.model.parameters()).device
        return {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in inputs.items()}

    def _inputs_minicpmv2(self, image, prompt, target, mask_mode):
        # Implementation note.
        # Implementation note.
        tok = self.tokenizer
        qn = self.model.config.query_num
        img_ph = f"{tok.im_start}{tok.unk_token * qn}{tok.im_end}"
        user_c = f"{img_ph}\n{prompt}"
        # MiniCPM-V-2 requires its native user-role control tag. Constructing
        # it from code points keeps source text and documentation English-only.
        user_role_tag = "<" + chr(29992) + chr(25143) + ">"
        full = f"{user_role_tag}{user_c}<AI>{target}"
        if tok.add_bos_token:
            input_ids = tok.encode(full)
        else:
            input_ids = [tok.bos_id] + tok.encode(full)
        input_ids = torch.tensor(input_ids, dtype=torch.int32)
        st = torch.where(input_ids == tok.im_start_id)[0] + 1
        en = torch.where(input_ids == tok.im_end_id)[0]
        image_bound = torch.hstack([st.unsqueeze(-1), en.unsqueeze(-1)])
        labels = input_ids.clone()
        if mask_mode == "assistant_only":
            prefix = f"{user_role_tag}{user_c}<AI>"
            if tok.add_bos_token:
                pids = tok.encode(prefix)
            else:
                pids = [tok.bos_id] + tok.encode(prefix)
            labels[: len(pids)] = -100
        data = {
            "input_ids": input_ids.unsqueeze(0),
            "image_bound": image_bound.unsqueeze(0),
            "position_ids": torch.arange(input_ids.shape[0]).unsqueeze(0),
            "pixel_values": [self._pixel_values_minicpm(image).unsqueeze(0)],
        }
        dev = next(self.model.parameters()).device
        data = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in data.items()}
        data["pixel_values"] = [p.to(dev) for p in data["pixel_values"]]
        return {"data": data, "labels": labels.unsqueeze(0).to(dev).long()}

    def _pixel_values_minicpm(self, image):
        from torchvision import transforms
        t = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        return t(image)

    def _inputs_deepseekvl2(self, image, prompt, target, mask_mode):
        msgs = [{"role": "user", "content": f"<image>\n{prompt}"},
                {"role": "assistant", "content": target}]
        text = self.processor.apply_chat_template(msgs, tokenize=False,
                                                  add_generation_prompt=False)
        enc = self.processor(text=[text], images=[image], return_tensors="pt")
        labels = enc["input_ids"].clone()
        if mask_mode == "assistant_only":
            pref = self.processor.apply_chat_template(
                [{"role": "user", "content": f"<image>\n{prompt}"}],
                tokenize=False, add_generation_prompt=True)
            p = self.processor(text=[pref], images=[image], return_tensors="pt")
            labels[:, :p["input_ids"].shape[1]] = -100
        enc["labels"] = labels
        dev = next(self.model.parameters()).device
        return {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in enc.items()}

    # Implementation note.
    def lm_forward(self, inputs: dict):
        out = self.model(**inputs)
        return out.loss, out.logits, inputs["labels"]

    def lm_backward(self, inputs: dict, selected: List[str]) -> Dict[str, torch.Tensor]:
        try:
            self.model.train(); self.model.zero_grad()
            with torch.amp.autocast("cuda"):
                out = self.model(**inputs)
            out.loss.backward()
            sel = set(selected)
            grads = {n: p.grad.detach().float()
                     for n, p in self.model.named_parameters()
                     if n in sel and p.grad is not None}
            self.model.zero_grad(); self.model.eval()
            return grads
        except Exception:
            self.model.zero_grad(); self.model.eval()
            return {}

    # Implementation note.
    def select(self, strategy: str) -> List[str]:
        k = KINDS[self.kind]
        return select_params(
            self.model, strategy,
            text_pattern=k["text_pattern"],
            vision_pattern=k["vision_pattern"],
        )

    # Implementation note.
    def tokenize(self, text: str):
        if self.tokenizer is None:
            return None
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def detokenize(self, ids) -> str:
        if self.tokenizer is None:
            return ""
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    # Implementation note.
    @torch.inference_mode()
    def generate(self, image, prompt: str, max_new_tokens: int = None) -> Optional[str]:
        try:
            inputs = self.build_lm_inputs(image, prompt, "", mask_mode="whole")
            if inputs is None:
                return None
            inputs.pop("labels", None)
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens
                                      or self.max_new_tokens)
            return self.processor.decode(out[0], skip_special_tokens=True)
        except Exception:
            return None
