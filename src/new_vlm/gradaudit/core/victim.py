"""GradAudit implementation documentation."""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import torch


class VictimModel(ABC):
    name: str = ""
    family: str = ""           # "mllm" | "clip"
    device: str = "cuda:0"

    @abstractmethod
    def load(self) -> None: ...

    # Implementation note.
    def build_lm_inputs(self, image, prompt: str, target: str,
                        mask_mode: str = "assistant_only", **kw) -> Optional[dict]:
        """GradAudit implementation documentation."""
        raise NotImplementedError

    def lm_forward(self, inputs: dict):
        """GradAudit implementation documentation."""
        raise NotImplementedError

    def lm_backward(self, inputs: dict, selected: List[str]) -> Dict[str, torch.Tensor]:
        """GradAudit implementation documentation."""
        raise NotImplementedError

    # Implementation note.
    def encode_image(self, image) -> torch.Tensor: raise NotImplementedError
    def encode_text(self, text: str) -> torch.Tensor: raise NotImplementedError
    def logit_scale(self) -> float: raise NotImplementedError

    # Implementation note.
    def tokenize(self, text: str):
        return None

    def detokenize(self, ids) -> str:
        return ""

    # Implementation note.
    def generate(self, image, prompt: str) -> Optional[str]:
        """GradAudit implementation documentation."""
        return None

    def contrastive_backward(self, images, captions,
                             selected: List[str]) -> Dict[str, torch.Tensor]:
        """GradAudit implementation documentation."""
        raise NotImplementedError

    # Implementation note.
    @abstractmethod
    def select(self, strategy: str) -> List[str]:
        """GradAudit implementation documentation."""

    def mark_trainable(self, selected: List[str]) -> None:
        names = set(selected)
        for n, p in self.model.named_parameters():
            p.requires_grad_(n in names)
