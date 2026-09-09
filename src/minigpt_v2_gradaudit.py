# =========================================================================
# See README.md for implementation and reproducibility details.
#
# See README.md for implementation and reproducibility details.
# See README.md for implementation and reproducibility details.
# See README.md for implementation and reproducibility details.
# See README.md for implementation and reproducibility details.
# See README.md for implementation and reproducibility details.
# See README.md for implementation and reproducibility details.
# See README.md for implementation and reproducibility details.
# See README.md for implementation and reproducibility details.
# See README.md for implementation and reproducibility details.
# See README.md for implementation and reproducibility details.
# See README.md for implementation and reproducibility details.
# See README.md for implementation and reproducibility details.
#        calibration(member-only ref) ✔  connector auto-detect ✔
#        vision/llama last-3 layers ✔  adaptive mask(tau+min_keep) ✔
#        probe scoring(mean) ✔  ROC / @5%FPR ✔  sample-level JSON ✔
#
# See README.md for implementation and reproducibility details.
#   python gradaudit_minigpt_server.py
#   python gradaudit_minigpt_server.py --output_dir /your/results/dir
# =========================================================================

import os, sys, gc, json, re, random, types, subprocess, argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

import numpy as np
from PIL import Image, UnidentifiedImageError
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from sklearn.metrics import (
    roc_auc_score, roc_curve, accuracy_score, confusion_matrix,
    precision_recall_fscore_support
)

import warnings
warnings.filterwarnings("ignore")

# =========================================================================
# See README.md for implementation and reproducibility details.
# =========================================================================
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR   = str(PACKAGE_ROOT / "data")

# See README.md for implementation and reproducibility details.
MODELS_DIR    = str(PACKAGE_ROOT / "models" / "minigpt_v2")
REPO_DIR      = Path(os.path.join(MODELS_DIR, "MiniGPT-4"))
CKPT_DIR      = Path(os.path.join(MODELS_DIR, "checkpoints"))
LLM_DIR       = Path(os.path.join(MODELS_DIR, "llm"))
HF_CACHE      = Path(os.path.join(MODELS_DIR, "hf_cache"))

CKPT_FILENAME = "minigptv2_checkpoint.pth"
LLM_REPO      = "meta-llama/Llama-2-7b-chat-hf"
MINIGPT_REPO_COMMIT = "d94738a7626ec43eba6c2cddf3cd2043f1a9689a"

# See README.md for implementation and reproducibility details.
MIA_DATA_DIR = os.path.join(DATA_DIR, "minigpt_coco_gptcap_mia")
DATASET_JSON = os.path.join(MIA_DATA_DIR, "dataset.json")
IMG_ROOT     = MIA_DATA_DIR

# See README.md for implementation and reproducibility details.
SEED             = 42
NUM_MEMBERS      = 1000
NUM_NONMEMBERS   = 1000

CALIB_MEM        = 200
CALIB_NON        = 200
PROBE_MEM        = 800
PROBE_NON        = 800

SENSITIVITY_TAU  = 0.10     # See README.md for implementation and reproducibility details.
MIN_KEEP_RATIO   = 0.20     # See README.md for implementation and reproducibility details.
MAX_PER_SPLIT    = 1000     # See README.md for implementation and reproducibility details.
CLEANUP_EVERY    = 25       # See README.md for implementation and reproducibility details.
BATCH_SIZE       = 8        # See README.md for implementation and reproducibility details.

# See README.md for implementation and reproducibility details.
IMAGE_SIZE = 448
PREPROCESS = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.48145466, 0.4578275,  0.40821073),
        std =(0.26862954, 0.26130258, 0.27577711),
    ),
])

PROMPT_RAW = "Describe the image in detail."

# =========================================================================
# 1) CLI
# =========================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="GradAudit MIA for MiniGPT-v2 (Server)")
    p.add_argument("--output_dir",   type=str,
                   default=str(PACKAGE_ROOT / "results" / "minigpt_v2"))
    p.add_argument("--dataset_json", type=str, default=DATASET_JSON)
    p.add_argument("--img_root",     type=str, default=IMG_ROOT)
    p.add_argument("--repo_dir",     type=str, default=str(REPO_DIR))
    p.add_argument("--ckpt_dir",     type=str, default=str(CKPT_DIR))
    p.add_argument("--llm_dir",      type=str, default=str(LLM_DIR))
    p.add_argument("--hf_cache",     type=str, default=str(HF_CACHE))
    return p.parse_args()

# =========================================================================
# See README.md for implementation and reproducibility details.
# =========================================================================
def setup_env():
    assert torch.cuda.is_available(), " GPU "
    random.seed(SEED); np.random.seed(SEED)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    torch.backends.cudnn.benchmark        = True
    torch.set_float32_matmul_precision("high")
    global AMP_DTYPE
    AMP_DTYPE = (torch.bfloat16
                 if torch.cuda.get_device_capability(0)[0] >= 8
                 else torch.float16)
    print(f"✅ Device: cuda:0  GPU: {torch.cuda.get_device_name(0)}  "
          f"dtype={AMP_DTYPE}")

AMP_DTYPE = torch.float16   # See README.md for implementation and reproducibility details.

# =========================================================================
# See README.md for implementation and reproducibility details.
# =========================================================================
def _run(cmd, quiet=False):
    cmd = [str(c) for c in cmd]
    print("▶", " ".join(cmd))
    subprocess.check_call(
        cmd, stdout=subprocess.DEVNULL if quiet else None)

def _banner(t):
    print("\n" + "=" * 70)
    print(t)
    print("=" * 70)

def _llm_ok(p: Path) -> bool:
    return (p / "config.json").exists() and (
        (p / "tokenizer.model").exists() or
        (p / "tokenizer.json").exists())

def setup_model_env(repo_dir: Path, ckpt_dir: Path,
                    llm_dir: Path, hf_cache: Path):
    """
    Reproduce the Colab setup steps using server-local paths.
    Existing repositories, checkpoints, and LLM assets are reused.
    """
    for d in [ckpt_dir, llm_dir, hf_cache]:
        d.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"]      = str(hf_cache)
    os.environ["HF_HUB_CACHE"] = str(hf_cache / "hub")

    # See README.md for implementation and reproducibility details.
    try:
        import peft, accelerate
        peft_ver = peft.__version__
    except ImportError:
        peft_ver = "0.0"

    if not (repo_dir / "minigpt4").exists() or peft_ver < "0.13":
        _banner("Step 1: Install dependencies")
        _run([sys.executable, "-m", "pip", "uninstall", "-y", "-q",
              "transformers", "peft", "accelerate", "bitsandbytes"],
             quiet=True)
        _run([sys.executable, "-m", "pip", "install", "-q",
              "huggingface_hub", "omegaconf", "pyyaml",
              "transformers==4.37.2", "accelerate==0.27.2",
              "peft==0.13.2", "bitsandbytes", "safetensors",
              "timm", "einops", "sentencepiece",
              "opencv-python-headless",
              "iopath", "fvcore", "yacs", "portalocker",
              "webdataset", "braceexpand", "decord"],
             quiet=True)

    # See README.md for implementation and reproducibility details.
    if not repo_dir.exists():
        _banner("Step 2: Clone MiniGPT-4 repo")
        _run(["git", "clone", "--quiet",
              "https://github.com/Vision-CAIR/MiniGPT-4.git",
              str(repo_dir)])
    _run(["git", "-C", str(repo_dir), "checkout", "--quiet",
          MINIGPT_REPO_COMMIT])

    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))

    # ── Step 3: Stub optional module ────────────────────────────────
    vg = types.ModuleType("visual_genome")
    vg.local = types.SimpleNamespace()
    sys.modules["visual_genome"] = vg

    # See README.md for implementation and reproducibility details.
    base_model_path = repo_dir / "minigpt4" / "models" / "base_model.py"
    if base_model_path.exists():
        txt = base_model_path.read_text(encoding="utf-8")
        if "PEFT compatibility shim (AUTO-INJECTED)" not in txt:
            _banner("Step 4: Patch base_model.py")
            pattern_block = r"(from\s+peft\s+import\s*\(\s*[\s\S]*?\))"
            m = re.search(pattern_block, txt)
            if m:
                block  = m.group(1)
                block2 = re.sub(
                    r"^\s*prepare_model_for_int8_training\s*,?\s*$",
                    "", block, flags=re.MULTILINE)
                block2 = re.sub(
                    r",\s*prepare_model_for_int8_training\s*(,)?",
                    lambda mm: "," if mm.group(1) else "", block2)
                block2 = re.sub(r",\s*,", ",", block2)
                txt = txt[:m.start()] + block2 + txt[m.end():]
            else:
                txt = re.sub(
                    r"(from\s+peft\s+import\s+.*)"
                    r"\bprepare_model_for_int8_training\b\s*,?\s*",
                    r"\1", txt)
            shim = (
                "# ===== PEFT compatibility shim (AUTO-INJECTED) =====\n"
                "try:\n"
                "    from peft import prepare_model_for_kbit_training"
                " as prepare_model_for_int8_training\n"
                "except Exception:\n"
                "    def prepare_model_for_int8_training(model, *args, **kwargs):\n"
                "        return model\n"
                "# =====================================================\n\n"
            )
            base_model_path.write_text(shim + txt, encoding="utf-8")
            print("✅ base_model.py patched")

    # See README.md for implementation and reproducibility details.
    llama_path = repo_dir / "minigpt4" / "models" / "modeling_llama.py"
    if llama_path.exists():
        existing = llama_path.read_text(encoding="utf-8")
        if "MiniGPT-v2 compatible LLaMA wrapper" not in existing:
            _banner("Step 5: Overwrite modeling_llama.py")
            llama_wrapper = r'''"""
MiniGPT-v2 compatible LLaMA wrapper (safe for transformers==4.37.2)
"""
import torch
from typing import Optional, List, Union, Tuple
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithPast

try:
    from transformers.models.llama.modeling_llama import LlamaForCausalLM as LlamaBase
except Exception:
    from transformers import LlamaForCausalLM as LlamaBase

class LlamaForCausalLM(LlamaBase):
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        reduction: str = "mean",
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask,
            position_ids=position_ids, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states, return_dict=True,
        )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states).float()
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss(reduction=reduction)
            loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1).to(shift_logits.device))
            if reduction == "none":
                loss = loss.view(logits.size(0), -1).mean(1)
        if not return_dict:
            out = (logits,) + outputs[1:]
            return (loss,) + out if loss is not None else out
        return CausalLMOutputWithPast(
            loss=loss, logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions)
'''
            llama_path.write_text(llama_wrapper, encoding="utf-8")
            print("✅ modeling_llama.py overwritten")

    # ── Step 6: Import minigpt4 modules ─────────────────────────────
    for k in list(sys.modules.keys()):
        if k.startswith("minigpt4"):
            sys.modules.pop(k, None)
    import minigpt4.models
    import minigpt4.processors
    import minigpt4.tasks
    import minigpt4.datasets.builders
    print("✅ minigpt4 imports OK")

    # See README.md for implementation and reproducibility details.
    ckpt_local_path = ckpt_dir / CKPT_FILENAME
    if not ckpt_local_path.exists() or ckpt_local_path.stat().st_size == 0:
        _banner("Step 7: Download MiniGPT-v2 checkpoint")
        from huggingface_hub import hf_hub_download
        ckpt_local_path = Path(hf_hub_download(
            repo_id="Vision-CAIR/MiniGPT-v2",
            filename=CKPT_FILENAME,
            repo_type="space",
            cache_dir=str(hf_cache),
            local_dir=str(ckpt_dir),
            local_dir_use_symlinks=False,
        ))
    print(f"✅ ckpt: {ckpt_local_path}  "
          f"({ckpt_local_path.stat().st_size/1024/1024:.1f} MB)")

    # See README.md for implementation and reproducibility details.
    llm_local_dir = llm_dir / "Llama-2-7b-chat-hf"
    if not _llm_ok(llm_local_dir):
        _banner("Step 8: Download Llama-2-7b-chat-hf")
        from huggingface_hub import snapshot_download
        snapshot_download(
            token=os.environ.get("HF_TOKEN"),
            repo_id=LLM_REPO,
            local_dir=str(llm_local_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
            cache_dir=str(hf_cache),
            ignore_patterns=["*.msgpack","*.h5","*.ot","*.md",".gitattributes"],
        )
    if not _llm_ok(llm_local_dir):
        raise RuntimeError(f"❌ LLM not ready: {llm_local_dir}")
    print(f"✅ LLM ready: {llm_local_dir}")

    # ── Step 9: Patch YAML configs ───────────────────────────────────
    _banner("Step 9: Patch configs")
    import yaml
    EVAL_CFG  = repo_dir / "eval_configs" / "minigptv2_eval.yaml"
    MODEL_CFG = (repo_dir / "minigpt4" / "configs" /
                 "models" / "minigpt_v2.yaml")

    eval_data = yaml.safe_load(EVAL_CFG.read_text())
    eval_data.setdefault("model", {})
    eval_data["model"]["ckpt"]          = str(ckpt_local_path)
    eval_data["model"]["low_resource"]  = False
    eval_data["model"]["device_8bit"]   = 0
    eval_data["model"]["load_in_8bit"]  = False
    eval_data["model"]["load_8bit"]     = False
    EVAL_CFG.write_text(yaml.safe_dump(eval_data, sort_keys=False))

    def _patch_llm(obj):
        if isinstance(obj, dict):  return {k: _patch_llm(v) for k, v in obj.items()}
        if isinstance(obj, list):  return [_patch_llm(v) for v in obj]
        if isinstance(obj, str):
            s = obj.lower()
            if any(t in s for t in ["meta-llama","llama-2","vicuna","lmsys","llama"]):
                return str(llm_local_dir)
        return obj

    model_data = yaml.safe_load(MODEL_CFG.read_text())
    model_data["model"] = _patch_llm(model_data["model"])
    MODEL_CFG.write_text(yaml.safe_dump(model_data, sort_keys=False))
    print("✅ configs patched")

    return EVAL_CFG, ckpt_local_path

# =========================================================================
# See README.md for implementation and reproducibility details.
# =========================================================================
def load_model(repo_dir: Path, eval_cfg: Path):
    from argparse import Namespace
    from minigpt4.common.config import Config
    from minigpt4.common.registry import registry

    device = torch.device("cuda:0")
    args   = Namespace(cfg_path=str(eval_cfg), options=[])
    cfg    = Config(args)
    model_cfg = cfg.model_cfg
    print(f"Model arch: {model_cfg.arch}")

    model_cls = registry.get_model_class(model_cfg.arch)
    assert model_cls is not None, f'❌ Model not registered: {model_cfg.arch}'
    model = model_cls.from_config(model_cfg)

    # See README.md for implementation and reproducibility details.
    # See README.md for implementation and reproducibility details.
    ckpt_path_in_yaml = model_cfg.get("ckpt", "")
    if ckpt_path_in_yaml and Path(ckpt_path_in_yaml).exists():
        ckpt  = torch.load(ckpt_path_in_yaml, map_location="cpu")
        state = ckpt.get("model", ckpt)
        msg   = model.load_state_dict(state, strict=False)
        print(f"✅ load_state_dict: missing={len(msg.missing_keys)}  "
              f"unexpected={len(msg.unexpected_keys)}")

    try:
        model.gradient_checkpointing_enable()
    except Exception:
        pass
    if hasattr(model, "config"):
        model.config.use_cache = False

    model = model.to(device).to(torch.bfloat16)
    model.eval()
    torch.cuda.empty_cache()
    print("✅ MiniGPT-v2 loaded on cuda:0")
    return model

# =========================================================================
# See README.md for implementation and reproducibility details.
# =========================================================================
class LocalCOCODataset(Dataset):
    def __init__(self, samples: List[Dict], img_root: str,
                 transform=None):
        self.samples   = samples
        self.img_root  = img_root
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        obj = self.samples[idx]
        rel = obj.get("image", "")
        p   = os.path.join(self.img_root, rel)
        try:
            img = Image.open(p).convert("RGB")
        except (UnidentifiedImageError, OSError, Exception):
            img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color="black")
        if self.transform:
            img = self.transform(img)
        cap = obj.get("caption", "")
        if isinstance(cap, list):
            cap = cap[0] if cap else ""
        return {"image": img, "caption": str(cap), "raw": obj}


def collate_fn(batch):
    images   = torch.stack([b["image"] for b in batch], dim=0)
    captions = [b["caption"] for b in batch]
    raws     = [b["raw"]     for b in batch]
    return {"image": images, "caption": captions, "raw": raws}


def load_and_split(dataset_json: str, img_root: str):
    with open(dataset_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    members_all    = [x for x in data if x["label"] == 1]
    nonmembers_all = [x for x in data if x["label"] == 0]
    rng = random.Random(SEED)
    members_all    = members_all[:NUM_MEMBERS]
    nonmembers_all = nonmembers_all[:NUM_NONMEMBERS]
    print(f"\n📦 Samples: members={len(members_all)} | "
          f"non-members={len(nonmembers_all)}")

    # See README.md for implementation and reproducibility details.
    calib_members = members_all[:CALIB_MEM]
    calib_nonmems = nonmembers_all[:CALIB_NON]
    probe_members = members_all[CALIB_MEM : CALIB_MEM + PROBE_MEM]
    probe_nonmems = nonmembers_all[CALIB_NON : CALIB_NON + PROBE_NON]

    def _make_loader(samples):
        ds = LocalCOCODataset(samples, img_root, transform=PREPROCESS)
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0, pin_memory=True,
                          collate_fn=collate_fn)

    print(f"Calibration: M={len(calib_members)}, N={len(calib_nonmems)} | "
          f"Probe: M={len(probe_members)}, N={len(probe_nonmems)}")

    return (_make_loader,
            calib_members, calib_nonmems,
            probe_members, probe_nonmems)

# =========================================================================
# See README.md for implementation and reproducibility details.
# =========================================================================
ATTN_PROJ_PAT  = re.compile(
    r"\.self_attn\.(q_proj|k_proj|v_proj|o_proj)\b", re.IGNORECASE)
ATTN_OTHER_PAT = re.compile(
    r"\b(attn|attention)\b.*\b(q_proj|k_proj|v_proj|o_proj|in_proj|out_proj)\b",
    re.IGNORECASE)

def _is_attention_proj(name: str) -> bool:
    ln = name.lower()
    return bool(ATTN_PROJ_PAT.search(ln) or ATTN_OTHER_PAT.search(ln))

CONNECTOR_KEYS = [
    "mm_projector","vision_proj","visual_proj",
    "vision_projection","visual_projection",
    "connector","bridge","mapping","align","adapter",
    "qformer","fusion","proj_to_llm","llm_proj","clip_proj",
]
VISION_HINT = re.compile(r"(visual|vision|eva|vit)", re.IGNORECASE)
LLAMA_HINT  = re.compile(
    r"(llama|language_model|llm|model)\.?.*layers\.\d+\.", re.IGNORECASE)

def _in_vision(n):  return bool(VISION_HINT.search(n))
def _in_llama(n):   return bool(LLAMA_HINT.search(n.lower()) or
                                (".layers." in n.lower() and
                                 ("llama" in n.lower() or
                                  "language_model" in n.lower())))

def _name_is_connector(n: str) -> bool:
    if _is_attention_proj(n): return False
    ln = n.lower()
    return any(k in ln for k in CONNECTOR_KEYS)

def find_connector_by_struct(model: nn.Module):
    cands = []
    for path, m in model.named_modules():
        if not path: continue
        pl = path.lower()
        if _in_vision(pl) or _in_llama(pl): continue
        is_linear   = isinstance(m, nn.Linear)
        is_seq      = isinstance(m, nn.Sequential)
        is_mlp_like = any(k in pl for k in [
            "projector","projection","connector","bridge",
            "mapping","align","adapter","qformer","fusion","mm"])
        if not (is_linear or is_seq or is_mlp_like): continue
        score = 0
        if is_linear:
            score += 3 if m.in_features != m.out_features else 1
        if is_seq:
            lins = [x for x in m.modules() if isinstance(x, nn.Linear)]
            score += 3 if len(lins) >= 2 else (1 if lins else 0)
            score += sum(1 for l in lins if l.in_features != l.out_features)
        if is_mlp_like: score += 2
        if score >= 3: cands.append((path, m, score))
    cands.sort(key=lambda x: x[2], reverse=True)
    return [(p, m) for p, m, _ in cands]

def _infer_max_layer(named_params, patterns):
    mx = -1
    for n, _ in named_params:
        for pat in patterns:
            mm = re.search(pat, n)
            if mm:
                try: mx = max(mx, int(mm.group(1)))
                except: pass
    return mx

def setup_grad_params(model):
    """ Colab  Section 3 & 4 """
    for _, p in model.named_parameters():
        p.requires_grad_(False)

    named_params = list(model.named_parameters())
    vision_pats  = [
        r"(?:visual|vision|eva|vit).*?(?:blocks|block|layers|layer)\.(\d+)\.",
        r"(?:visual|vision|eva|vit)\.(?:blocks|block|layers|layer)\.(\d+)\.",
    ]
    llama_pats   = [
        r"(?:llama|llm|language_model|model)\.?.*layers\.(\d+)\.",
        r"(?:llama_model|llama)\.?.*layers\.(\d+)\.",
    ]
    max_v = _infer_max_layer(named_params, vision_pats)
    max_l = _infer_max_layer(named_params, llama_pats)
    v_start = max(0, max_v - 2) if max_v >= 0 else None
    l_start = max(0, max_l - 2) if max_l >= 0 else None
    print(f"🔎 Vision: max={max_v}, last-3 start={v_start}")
    print(f"🔎 LLaMA:  max={max_l}, last-3 start={l_start}")

    def _is_vision_last3(n):
        if v_start is None: return False
        if not any(k in n.lower() for k in ["visual","vision","eva","vit"]):
            return False
        for pat in vision_pats:
            mm = re.search(pat, n)
            if mm:
                try: return int(mm.group(1)) >= v_start
                except: return False
        return False

    def _is_llama_last3(n):
        if l_start is None: return False
        for pat in llama_pats:
            mm = re.search(pat, n)
            if mm:
                try: return int(mm.group(1)) >= l_start
                except: return False
        return False

    # connector names
    conn_by_name  = {n for n, _ in named_params if _name_is_connector(n)}
    mod_paths     = {p for p, _ in find_connector_by_struct(model)}
    conn_by_struct= {n for n, _ in named_params
                     if any(n.startswith(p+".") for p in mod_paths)
                     and not _is_attention_proj(n)}
    conn_names    = conn_by_name | conn_by_struct

    print(f"Connector by-name={len(conn_by_name)} "
          f"by-struct={len(conn_by_struct)}")

    enabled, groups = [], {"visual":[], "text":[], "connector":[]}
    name_to_param   = dict(named_params)

    for n, p in named_params:
        if p.ndim < 2: continue
        grp = None
        if n in conn_names:      grp = "connector"
        elif _is_vision_last3(n): grp = "visual"
        elif _is_llama_last3(n):  grp = "text"
        if grp:
            p.requires_grad_(True)
            enabled.append(n)
            groups[grp].append((n, tuple(p.shape)))

    print(f"🎯 : {len(enabled)}")
    for g, items in groups.items():
        print(f"  {g.upper():<10}: {len(items)}")
    return sorted(enabled), name_to_param

# =========================================================================
# See README.md for implementation and reproducibility details.
# =========================================================================
def ensure_placeholder(p: str) -> str:
    p = (p or "").strip()
    return p if "<ImageHere>" in p else "<ImageHere> " + p

PROMPT      = ensure_placeholder(PROMPT_RAW)
PROMPT_LIST = [PROMPT]

def _make_samples(image_1, caption_1):
    cap = caption_1 if (isinstance(caption_1, str)
                        and caption_1.strip()) else " "
    return {"image": image_1, "text_input": PROMPT_LIST, "answer": [cap]}

def _forward_loss(model, samples):
    try:
        out = model.forward(samples, reduction="mean")
    except TypeError:
        out = model(samples, reduction="mean")
    if isinstance(out, dict) and "loss" in out: return out["loss"]
    if hasattr(out, "loss"):                    return out.loss
    if torch.is_tensor(out):                    return out
    raise RuntimeError("❌ Cannot find loss in model output.")

# =========================================================================
# See README.md for implementation and reproducibility details.
# =========================================================================
_step_counter = 0

def compute_grad_dict_gpu(model, enabled_names, name_to_param,
                          image_1, caption_1):
    global _step_counter
    device = next(model.parameters()).device
    model.zero_grad(set_to_none=True)
    samples = _make_samples(image_1, caption_1)

    try:
        with torch.amp.autocast(device_type="cuda", dtype=AMP_DTYPE):
            loss = _forward_loss(model, samples)
        if (not torch.is_tensor(loss) or
                torch.isnan(loss) or torch.isinf(loss)):
            return None
        loss.backward()

        grad_dict = {}
        for name in enabled_names:
            p = name_to_param[name]
            g = p.grad
            if g is None: continue
            g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
            grad_dict[name] = g.detach().half()

        if not grad_dict: return None

        _step_counter += 1
        if (_step_counter % CLEANUP_EVERY) == 0:
            gc.collect()
        return grad_dict

    except RuntimeError as e:
        import traceback
        print(f"\n🔴 [DEBUG] compute_grad_dict_gpu RuntimeError: {e}")
        traceback.print_exc()
        if "out of memory" in str(e).lower():
            model.zero_grad(set_to_none=True)
            gc.collect(); torch.cuda.empty_cache()
        return None
    except Exception as e:
        import traceback
        print(f"\n🔴 [DEBUG] compute_grad_dict_gpu Exception: {e}")
        traceback.print_exc()
        return None
    finally:
        model.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()

# =========================================================================
# See README.md for implementation and reproducibility details.
# =========================================================================
def row_col_cosine_gpu(grad_gpu, ref_gpu):
    if grad_gpu.ndim < 2:
        flat = F.cosine_similarity(
            grad_gpu.flatten().unsqueeze(0),
            ref_gpu.flatten().unsqueeze(0), dim=1)
        flat = torch.nan_to_num(flat, nan=0.0)
        return flat, flat
    row = torch.nan_to_num(F.cosine_similarity(grad_gpu, ref_gpu, dim=1),    nan=0.0)
    col = torch.nan_to_num(F.cosine_similarity(grad_gpu.T, ref_gpu.T, dim=1),nan=0.0)
    return row, col

# =========================================================================
# See README.md for implementation and reproducibility details.
# =========================================================================
def build_reference_gradient(model, enabled_names, name_to_param,
                              loader, device, tag="member", take=200):
    ref_sum_cpu = {}
    ref_count   = 0
    seen        = 0
    print(f"\n📌  ({tag}, target={take}) ...")

    for batch in tqdm(loader, desc=f"ref-{tag}"):
        imgs = batch["image"].to(device, non_blocking=True)
        caps = batch["caption"]
        bsz  = imgs.size(0)
        for i in range(bsz):
            if ref_count >= take: break
            gd = compute_grad_dict_gpu(
                model, enabled_names, name_to_param,
                imgs[i:i+1], caps[i])
            if gd is None: continue
            ref_count += 1
            if not ref_sum_cpu:
                ref_sum_cpu = {k: v.float().cpu().clone()
                               for k, v in gd.items()}
            else:
                for k in list(ref_sum_cpu.keys()):
                    if (k in gd and
                            ref_sum_cpu[k].shape == gd[k].shape):
                        ref_sum_cpu[k].add_(gd[k].float().cpu())
                    else:
                        ref_sum_cpu.pop(k, None)
        seen += bsz
        if ref_count >= take or seen >= MAX_PER_SPLIT: break

    if ref_count == 0:
        raise RuntimeError(f"❌ : {tag}")

    # The published main-table run quantized the averaged reference prototype
    # to FP16 before row/column cosine similarity was evaluated.
    ref_avg_gpu = {k: (v / ref_count).half().to(device)
                   for k, v in ref_sum_cpu.items()}
    print(f"✅ : {tag}, ={ref_count}, ={len(ref_avg_gpu)}")
    return ref_avg_gpu

# =========================================================================
# See README.md for implementation and reproducibility details.
# =========================================================================
def compute_avg_rowcol_similarity(model, enabled_names, name_to_param,
                                   loader, ref_grads_gpu, device,
                                   tag="member", take=200):
    acc_row_gpu = {}
    acc_col_gpu = {}
    ok   = 0
    seen = 0
    print(f"\n📌  ({tag}, target={take}) ...")

    for batch in tqdm(loader, desc=f"sim-{tag}"):
        imgs = batch["image"].to(device, non_blocking=True)
        caps = batch["caption"]
        bsz  = imgs.size(0)
        for i in range(bsz):
            if ok >= take: break
            gd = compute_grad_dict_gpu(
                model, enabled_names, name_to_param,
                imgs[i:i+1], caps[i])
            if gd is None: continue
            ok += 1
            for name, g in gd.items():
                ref = ref_grads_gpu.get(name)
                if ref is None or ref.shape != g.shape: continue
                rs, cs = row_col_cosine_gpu(g, ref)
                if name not in acc_row_gpu:
                    acc_row_gpu[name] = rs.float()
                    acc_col_gpu[name] = cs.float()
                else:
                    if acc_row_gpu[name].shape == rs.shape:
                        acc_row_gpu[name].add_(rs.float())
                    if acc_col_gpu[name].shape == cs.shape:
                        acc_col_gpu[name].add_(cs.float())
        seen += bsz
        if ok >= take or seen >= MAX_PER_SPLIT: break

    if ok == 0:
        print(f"⚠️ {tag}: ok=0, ")
        return {}, {}

    avg_r = {k: v / ok for k, v in acc_row_gpu.items()}
    avg_c = {k: v / ok for k, v in acc_col_gpu.items()}
    print(f"✅ : {tag}, ok={ok}, ={len(avg_r)}")
    return avg_r, avg_c

# =========================================================================
# See README.md for implementation and reproducibility details.
# =========================================================================
def build_adaptive_masks(mem_r, mem_c, non_r, non_c,
                         tau=SENSITIVITY_TAU,
                         min_keep=MIN_KEEP_RATIO):
    row_masks, col_masks, total = {}, {}, 0
    total_dims = 0
    print(f"\n📌  (tau={tau}, min_keep={min_keep}) ...")

    for name in mem_r:
        if name not in non_r or name not in mem_c or name not in non_c:
            continue
        row_gap = mem_r[name] - non_r[name]
        col_gap = mem_c[name] - non_c[name]

        def _make_mask(gap):
            sz = gap.numel(); total_dims_add = sz
            tau_mask = (gap > tau)
            if tau_mask.sum().item() < sz * min_keep:
                k = max(1, int(sz * min_keep))
                _, idx = torch.topk(gap.flatten(), k)
                m = torch.zeros_like(gap, dtype=torch.bool)
                m.view(-1)[idx] = True
                return m, total_dims_add
            return tau_mask, total_dims_add

        rm, rd = _make_mask(row_gap)
        cm, cd = _make_mask(col_gap)
        row_masks[name] = rm
        col_masks[name] = cm
        total_dims += rd + cd
        total += int(rm.sum().item() + cm.sum().item())

    ratio = total / total_dims if total_dims > 0 else 0
    print(f"✅ : {total}/{total_dims} ({ratio*100:.1f}%)")
    return row_masks, col_masks, total

# =========================================================================
# See README.md for implementation and reproducibility details.
# =========================================================================
def probe_dataset(model, enabled_names, name_to_param,
                  loader, ref_grads_gpu, row_masks, col_masks,
                  device, tag="member", take=800,
                  raw_samples: Optional[List[Dict]] = None):
    scores  = []
    records = []
    seen    = 0
    print(f"\n🚀  {tag} (target={take}) ...")

    for batch in tqdm(loader, desc=f"probe-{tag}"):
        imgs = batch["image"].to(device, non_blocking=True)
        caps = batch["caption"]
        raws = batch["raw"]
        bsz  = imgs.size(0)

        for i in range(bsz):
            if seen >= take: break
            gd = compute_grad_dict_gpu(
                model, enabled_names, name_to_param,
                imgs[i:i+1], caps[i])

            if gd is None:
                s = 0.0
            else:
                tot_sum = None
                tot_cnt = 0
                for name, g in gd.items():
                    ref  = ref_grads_gpu.get(name)
                    rmsk = row_masks.get(name)
                    cmsk = col_masks.get(name)
                    if ref is None or rmsk is None or cmsk is None: continue
                    if ref.shape != g.shape: continue
                    rs, cs = row_col_cosine_gpu(g, ref)
                    for v, msk in [(rs, rmsk), (cs, cmsk)]:
                        if msk.any():
                            vals = v[msk]
                            tot_cnt += vals.numel()
                            tot_sum  = (vals.sum() if tot_sum is None
                                        else tot_sum + vals.sum())
                s = float((tot_sum / tot_cnt).item()) if tot_cnt > 0 else 0.0

            scores.append(s)
            lbl = 1 if tag == "member" else 0
            records.append({
                "image":   str(raws[i].get("image",   "?")),
                "caption": str(raws[i].get("caption", ""))[:120],
                "score":   float(s),
                "label":   lbl,
                "split":   f"probe_{tag}"
            })
            seen += 1
            if (seen % CLEANUP_EVERY) == 0: gc.collect()

        if seen >= take or seen >= MAX_PER_SPLIT: break

    print(f"✅ : {tag}, ={len(scores)}")
    return scores, records

# =========================================================================
# See README.md for implementation and reproducibility details.
# =========================================================================
def compute_metrics(scores: np.ndarray, labels: np.ndarray) -> Dict:
    if np.all(scores == scores[0]):
        return {"auc": 0.5, "best_threshold": float(np.median(scores)),
                "accuracy": 0.5, "confusion_matrix": None,
                "fpr5_metrics": None,
                "roc_data": {"fpr": [], "tpr": [], "thresholds": []}}

    auc = roc_auc_score(labels, scores)
    fpr_arr, tpr_arr, thr_arr = roc_curve(labels, scores)
    j        = int(np.argmax(tpr_arr - fpr_arr))
    best_thr = float(thr_arr[j])
    pred     = (scores >= best_thr).astype(int)
    acc      = float(accuracy_score(labels, pred))
    cm       = confusion_matrix(labels, pred).tolist()

    nonmask      = (labels == 0)
    fpr5_metrics = None
    if nonmask.sum() > 0:
        thr_5  = float(np.quantile(scores[nonmask], 0.95))
        pred_5 = (scores >= thr_5).astype(int)
        cm_5   = confusion_matrix(labels, pred_5)
        tn, fp, fn, tp = cm_5.ravel()
        prec, rec, f1, _ = precision_recall_fscore_support(
            labels, pred_5, average="binary", zero_division=0)
        fpr5_metrics = {
            "threshold":        thr_5,
            "actual_fpr":       float(fp / (fp + tn + 1e-12)),
            "tpr":              float(tp / (tp + fn + 1e-12)),
            "precision":        float(prec),
            "recall":           float(rec),
            "f1":               float(f1),
            "accuracy":         float(accuracy_score(labels, pred_5)),
            "confusion_matrix": cm_5.tolist()
        }

    return {
        "auc":              float(auc),
        "best_threshold":   float(best_thr),
        "accuracy":         acc,
        "confusion_matrix": cm,
        "fpr5_metrics":     fpr5_metrics,
        "roc_data": {
            "fpr":        fpr_arr.tolist(),
            "tpr":        tpr_arr.tolist(),
            "thresholds": thr_arr.tolist()
        }
    }

# =========================================================================
# See README.md for implementation and reproducibility details.
# =========================================================================
def save_results(output_dir: str, metrics: Dict,
                 sample_records: List[Dict], total_crit: int,
                 enabled_len: int):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary = {
        "timestamp": ts,
        "config": {
            "model":            "MiniGPT-v2",
            "seed":             SEED,
            "calib_mem":        CALIB_MEM,
            "calib_non":        CALIB_NON,
            "probe_mem":        PROBE_MEM,
            "probe_non":        PROBE_NON,
            "sensitivity_tau":  SENSITIVITY_TAU,
            "min_keep_ratio":   MIN_KEEP_RATIO,
            "score_agg":        "mean",
            "mask_type":        "adaptive(tau+min_keep)",
            "ref_grad_source":  "member_only",
            "enabled_params":   enabled_len,
            "critical_dims":    total_crit,
        },
        "metrics": {k: v for k, v in metrics.items() if k != "roc_data"}
    }
    p1 = os.path.join(output_dir, f"results_summary_{ts}.json")
    with open(p1, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    p2 = os.path.join(output_dir, f"sample_scores_{ts}.json")
    with open(p2, "w", encoding="utf-8") as f:
        json.dump(sample_records, f, indent=2, ensure_ascii=False)

    p3 = os.path.join(output_dir, f"roc_data_{ts}.json")
    with open(p3, "w", encoding="utf-8") as f:
        json.dump(metrics.get("roc_data", {}), f, indent=2, ensure_ascii=False)

    print("\n💾 Results saved:")
    print(f"   summary       → {p1}")
    print(f"   sample_scores → {p2}")
    print(f"   roc_data      → {p3}")

# =========================================================================
# See README.md for implementation and reproducibility details.
# =========================================================================
def print_metrics(metrics: Dict, total_crit: int):
    m = metrics
    print("\n" + "=" * 70)
    print("📊 GradSafe MIA  (MiniGPT-v2)")
    print("=" * 70)
    print(f" (tau={SENSITIVITY_TAU}, min_keep={MIN_KEEP_RATIO}): "
          f"{total_crit}")
    print(f"AUC      = {m.get('auc','N/A')}")
    print(f"  = {m.get('best_threshold','N/A')}")
    print(f"    = {m.get('accuracy','N/A')}")
    print("Confusion matrix [[TN FP][FN TP]]:")
    print(np.array(m.get("confusion_matrix", [[0,0],[0,0]])))

    fp5 = m.get("fpr5_metrics")
    print("\n------ Metrics @ 5% FPR ------")
    if fp5:
        print(f"Threshold  = {fp5['threshold']:.6f}")
        print(f"Actual FPR = {fp5['actual_fpr']*100:.2f}%")
        print(f"TPR        = {fp5['tpr']*100:.2f}%")
        print(f"Precision  = {fp5['precision']*100:.2f}%")
        print(f"F1         = {fp5['f1']:.4f}")
        print(f"Accuracy   = {fp5['accuracy']*100:.2f}%")
        print("Confusion @5%FPR [[TN FP][FN TP]]:")
        print(np.array(fp5["confusion_matrix"]))
    else:
        print("⚠️  @5% FPR ")

# =========================================================================
# See README.md for implementation and reproducibility details.
# =========================================================================
def main():
    args = parse_args()
    setup_env()

    repo_dir  = Path(args.repo_dir)
    ckpt_dir  = Path(args.ckpt_dir)
    llm_dir   = Path(args.llm_dir)
    hf_cache  = Path(args.hf_cache)

    # See README.md for implementation and reproducibility details.
    eval_cfg, _ = setup_model_env(repo_dir, ckpt_dir, llm_dir, hf_cache)
    model       = load_model(repo_dir, eval_cfg)
    device      = next(model.parameters()).device

    # See README.md for implementation and reproducibility details.
    (_make_loader,
     calib_members, calib_nonmems,
     probe_members, probe_nonmems) = load_and_split(
         args.dataset_json, args.img_root)

    # See README.md for implementation and reproducibility details.
    enabled_names, name_to_param = setup_grad_params(model)
    if not enabled_names:
        print("❌ No gradient parameters were selected.")
        sys.exit(1)

    try:
        # See README.md for implementation and reproducibility details.
        print("\n[Step 1/4] Reference gradient (member-only)...")
        ref_grads_gpu = build_reference_gradient(
            model, enabled_names, name_to_param,
            _make_loader(calib_members), device, "member", CALIB_MEM)

        # See README.md for implementation and reproducibility details.
        print("\n[Step 2/4] …")
        mr, mc = compute_avg_rowcol_similarity(
            model, enabled_names, name_to_param,
            _make_loader(calib_members), ref_grads_gpu, device, "member", CALIB_MEM)
        nr, nc = compute_avg_rowcol_similarity(
            model, enabled_names, name_to_param,
            _make_loader(calib_nonmems), ref_grads_gpu, device, "nonmember", CALIB_NON)

        row_masks, col_masks, total_crit = build_adaptive_masks(
            mr, mc, nr, nc, SENSITIVITY_TAU, MIN_KEEP_RATIO)

        if total_crit == 0:
            print("❌ No sensitive dimensions were selected.")
            sys.exit(1)

        # See README.md for implementation and reproducibility details.
        print("\n[Step 3/4] …")
        mem_scores, mem_records = probe_dataset(
            model, enabled_names, name_to_param,
            _make_loader(probe_members), ref_grads_gpu, row_masks, col_masks,
            device, "member", PROBE_MEM)
        non_scores, non_records = probe_dataset(
            model, enabled_names, name_to_param,
            _make_loader(probe_nonmems), ref_grads_gpu, row_masks, col_masks,
            device, "nonmember", PROBE_NON)

        # See README.md for implementation and reproducibility details.
        print("\n[Step 4/4]  & …")
        scores  = np.array(mem_scores + non_scores, dtype=np.float64)
        labels  = np.array([1]*len(mem_scores) + [0]*len(non_scores),
                            dtype=np.int64)
        records = mem_records + non_records

        metrics = compute_metrics(scores, labels)
        print_metrics(metrics, total_crit)
        save_results(args.output_dir, metrics, records,
                     total_crit, len(enabled_names))

    except RuntimeError as e:
        print(f"\n❌ : {e}")
        sys.exit(1)
    except Exception as e:
        import traceback
        print(f"\n❌ : {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
