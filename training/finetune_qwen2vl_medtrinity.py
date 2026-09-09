# =========================================================================
# Implementation note.
# Implementation note.
#               (LLaMA-Factory, epoch=5, rank=64, lr=1e-4, batch=4)
# Implementation note.
# Implementation note.
# Implementation note.
#
# Implementation note.
# Implementation note.
#               - SEED = 20251012
# Implementation note.
# Implementation note.
# Implementation note.
# Implementation note.
# Implementation note.
# Implementation note.
#
# Implementation note.
# Implementation note.
# Implementation note.
# Implementation note.
# =========================================================================

import os, sys, json, random, subprocess, re, gc, shutil, warnings, argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

import numpy as np
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

import torch
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, confusion_matrix

warnings.filterwarnings("ignore")

# =========================================================================
# Implementation note.
# =========================================================================
BASE_DIR  = str(Path(__file__).resolve().parents[1])
DATA_DIR  = os.path.join(BASE_DIR, "data")

MIA_DATA_DIR      = os.path.join(DATA_DIR, "frozen_dataset/medtrinity_mia")
MEMBER_5K_DIR     = os.path.join(MIA_DATA_DIR, "member_5k")
NONMEMBER_5K_DIR  = os.path.join(MIA_DATA_DIR, "nonmember_5k")

MEMBER_EVAL_JSON    = os.path.join(MIA_DATA_DIR, "member_eval_1000.json")
NONMEMBER_EVAL_JSON = os.path.join(MIA_DATA_DIR, "nonmember_eval_1000.json")

MODELS_DIR    = os.path.join(BASE_DIR, "models")
BASE_MODEL_2B = os.path.join(MODELS_DIR, "Qwen2-VL-2B-Instruct")
LLAMA_REPO    = os.path.join(MODELS_DIR, "LLaMA-Factory")
HF_CACHE_DIR  = os.path.join(MODELS_DIR, "hf_cache")

WORK_DIR = os.path.join(BASE_DIR, "experiment_workspace")

# Implementation note.
TRAIN_SEED        = 42
TRAIN_MAX_SAMPLES = 5000
TRAIN_EPOCHS      = 5.0
SAVE_TOTAL_LIMIT  = 30
LORA_RANK         = 64
LORA_ALPHA        = 128

# Implementation note.
ATTACK_SEED    = 20251012          # Implementation note.
CALIB_MEM      = 200
CALIB_NON      = 200
PROBE_MEM      = 800
PROBE_NON      = 800
QUANTILE_Q     = 0.80              # Implementation note.
TOP_K_FALLBACK = 16               # Implementation note.
DTYPE          = torch.bfloat16 if torch.cuda.is_available() else torch.float32
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

# Implementation note.
MEMBER_JSON    = MEMBER_EVAL_JSON
NONMEMBER_JSON = NONMEMBER_EVAL_JSON
MEMBER_ROOT    = MEMBER_5K_DIR
NONMEMBER_ROOT = NONMEMBER_5K_DIR

RESULT_DIR = os.path.join(BASE_DIR, "results/qwen_med/gradaudit")
OUT_CSV    = os.path.join(RESULT_DIR, "mia_gradsafe_qwen2vl_lora_eval1000.csv")

CONFIG = {
    "id":         "medtrinity_1k_2b",
    "train_size": 1000,
    "model":      "2B",
    "ref_size":   200,
}

# Implementation note.
PROMPT_USER_EN = (
    "<image> As a medical vision-language model, please analyze the image and "
    "generate a concise, clinically sound diagnostic summary."
)

# Implementation note.
PROMPT_EN = (
    "As a medical vision-language model, describe this medical image and "
    "provide a professional diagnostic impression."
)

os.environ["TOKENIZERS_PARALLELISM"]            = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"]           = "max_split_size_mb:128"
os.environ["MPLBACKEND"]                         = "Agg"


# =========================================================================
# 1) CLI
# =========================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Train (code-1) then GradSafe attack (code-2) for Qwen2-VL MedTrinity")
    p.add_argument("--fresh",       action="store_true",
                   help="Remove old adapter outputs and train from scratch")
    p.add_argument("--skip_train",  action="store_true",
                   help="Skip training and evaluate the existing adapter")
    p.add_argument("--config_id",   type=str, default=CONFIG["id"])
    p.add_argument("--base_model",  type=str, default=BASE_MODEL_2B)
    p.add_argument("--llama_repo",  type=str, default=LLAMA_REPO)
    p.add_argument("--out_csv",     type=str, default=OUT_CSV)
    p.add_argument("--out_dir",     type=str, default=RESULT_DIR)
    return p.parse_args()


# =========================================================================
# Implementation note.
# =========================================================================
def setup_train_env():
    assert torch.cuda.is_available(), "A GPU is required"
    random.seed(TRAIN_SEED); np.random.seed(TRAIN_SEED)
    torch.manual_seed(TRAIN_SEED); torch.cuda.manual_seed_all(TRAIN_SEED)
    print(f"Device: cuda:0  GPU: {torch.cuda.get_device_name(0)}")

def set_all_seeds(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================================================================
# Implementation note.
# =========================================================================

# Implementation note.
def _convert_simple_to_sharegpt(records):
    converted = []
    for r in records:
        if "messages" in r and "images" in r:
            converted.append(r)
            continue
        img = r.get("image") or (r.get("images", [None])[0] if r.get("images") else None)
        caption = r.get("caption", "")
        if img is None:
            continue
        converted.append({
            "images": [img],
            "messages": [
                {"role": "user", "content": PROMPT_USER_EN},
                {"role": "assistant", "content": caption},
            ]
        })
    return converted

def _assert_fmt(sample, name):
    assert ("images" in sample and isinstance(sample["images"], list)
            and len(sample["images"]) > 0), f"{name}: missing images"
    assert ("messages" in sample and isinstance(sample["messages"], list)
            and len(sample["messages"]) >= 2), f"{name}: missing messages"

def prepare_training_data(config: Dict, work_dir: str) -> Dict[str, str]:
    print(f"\n{'='*70}")
    print("Section 1: Prepare flat ShareGPT training data")
    print(f"{'='*70}")

    mem_eval_raw = _load_json(MEMBER_EVAL_JSON)
    if isinstance(mem_eval_raw, dict) and "data" in mem_eval_raw:
        mem_eval_raw = mem_eval_raw["data"]
    mem_eval = _convert_simple_to_sharegpt(mem_eval_raw)
    _assert_fmt(mem_eval[0], "mem_eval[0]")

    cid  = config["id"]
    wdir = os.path.join(work_dir, cid)
    Path(f"{wdir}/train_data").mkdir(parents=True, exist_ok=True)

    flat_train_json = f"{wdir}/train_data/mllm_data_eval_1000_flat.json"
    with open(flat_train_json, "w", encoding="utf-8") as f:
        json.dump(mem_eval, f, ensure_ascii=False)

    print(f"Training data (member_eval_1000.json): {len(mem_eval)} records ->  {flat_train_json}")
    return {"flat_train_json": flat_train_json}


# Implementation note.
def _latest_ckpt(root: str) -> Optional[str]:
    p = Path(root)
    if not p.exists(): return None
    cands = []
    for d in p.glob("checkpoint-*"):
        m = re.search(r"checkpoint-(\d+)$", d.name)
        if m: cands.append((int(m.group(1)), str(d)))
    if not cands: return None
    cands.sort(key=lambda x: x[0], reverse=True)
    return cands[0][1]


def _find_adapters(root: str) -> list:
    hits = []
    for p in Path(root).rglob("adapter_config.json"):
        d = str(p.parent)
        m = re.search(r"checkpoint[-_]?(\d+)", d)
        step = int(m.group(1)) if m else -1
        try:   mtime = os.path.getmtime(d)
        except Exception: mtime = 0.0
        hits.append((step, mtime, d))
    if not hits and (Path(root)/"adapter_config.json").is_file():
        hits.append((10**9, os.path.getmtime(root), root))
    return sorted(hits, key=lambda x: (x[0], x[1]), reverse=True)


def clean_previous_outputs(config: Dict, work_dir: str):
    """GradAudit implementation documentation."""
    cid         = config["id"]
    output_dir  = os.path.join(work_dir, cid, "lora_model")
    adapter_dir = os.path.join(work_dir, cid, "adapter_final")
    for d in [output_dir, adapter_dir]:
        if os.path.isdir(d):
            print(f"[--fresh] Removing previous output: {d}")
            shutil.rmtree(d)


def register_dataset(config_id: str, flat_json: str, llama_repo: str) -> str:
    assert os.path.isdir(llama_repo), f"LLaMA-Factory not found: {llama_repo}"
    os.chdir(llama_repo)
    info_path = "data/dataset_info.json"
    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)
    key = f"mllm_member_eval_1000_{config_id}"
    info[key] = {
        "file_name": flat_json,
        "formatting": "sharegpt",
        "columns": {"messages": "messages", "images": "images"},
        "tags": {"role_tag": "role", "content_tag": "content",
                 "user_tag": "user", "assistant_tag": "assistant"}
    }
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(f"Registered dataset_info.json entry: {key}")
    return key


def train_lora(config: Dict, flat_train_json: str, work_dir: str,
               base_model: str, llama_repo: str,
               skip_train: bool = False) -> Dict[str, str]:
    print(f"\n{'='*70}")
    print(f"Section 2: LoRA training (epoch={TRAIN_EPOCHS}, rank={LORA_RANK})")
    print(f"{'='*70}")

    cid         = config["id"]
    output_dir  = os.path.join(work_dir, cid, "lora_model")
    log_path    = os.path.join(work_dir, cid, "train.log")
    adapter_dir = os.path.join(work_dir, cid, "adapter_final")

    if (skip_train or (
            os.path.exists(f"{adapter_dir}/adapter_config.json") and (
            os.path.exists(f"{adapter_dir}/adapter_model.safetensors") or
            os.path.exists(f"{adapter_dir}/adapter_model.bin")))):
        print(f"Adapter exists; skipping training: {adapter_dir}")
        return {"base_model": base_model, "lora_dir": adapter_dir, "log_path": None}

    dataset_key = register_dataset(cid, flat_train_json, llama_repo)
    resume_dir  = _latest_ckpt(output_dir)

    if resume_dir:
        print(f"Resuming checkpoint: {resume_dir}")
    else:
        print("Training from scratch")
        if os.path.isdir(output_dir):
            shutil.rmtree(output_dir)

    script = f"""#!/usr/bin/env bash
set -euo pipefail
cd "{llama_repo}"

python -m llamafactory.cli train \\
  --stage sft \\
  --do_train \\
  --model_name_or_path "{base_model}" \\
  --cache_dir "{HF_CACHE_DIR}" \\
  --trust_remote_code \\
  --finetuning_type lora \\
  --template qwen2_vl \\
  --media_dir "{MEMBER_5K_DIR}" \\
  --dataset_dir data \\
  --dataset {dataset_key} \\
  --cutoff_len 2048 \\
  --learning_rate 1e-4 \\
  --num_train_epochs {TRAIN_EPOCHS} \\
  --max_samples {TRAIN_MAX_SAMPLES} \\
  --per_device_train_batch_size 4 \\
  --gradient_accumulation_steps 4 \\
  --lr_scheduler_type cosine \\
  --max_grad_norm 1.0 \\
  --logging_steps 5 \\
  --save_strategy steps \\
  --save_steps 100 \\
  --save_total_limit {SAVE_TOTAL_LIMIT} \\
  --warmup_steps 0 \\
  --optim adamw_torch \\
  --report_to none \\
  --output_dir "{output_dir}" \\
  --overwrite_output_dir \\
  --bf16 \\
  --plot_loss False \\
  --flash_attn sdpa \\
  --dataloader_num_workers 0 \\
  --no_dataloader_pin_memory \\
  --torch_empty_cache_steps 20 \\
  --lora_rank {LORA_RANK} \\
  --lora_alpha {LORA_ALPHA} \\
  --lora_dropout 0.0 \\
  --lora_target all \\
  --gradient_checkpointing \\
  {"--resume_from_checkpoint " + resume_dir if resume_dir else ""}
"""
    sh_path = os.path.join(work_dir, f"run_lora_{cid}.sh")
    Path(sh_path).write_text(script)
    os.chmod(sh_path, 0o755)

    print("Starting LLaMA-Factory training (1,000 records x 5 epochs)...")
    with open(log_path, "wb") as logf:
        proc = subprocess.Popen(
            ["bash", "-lc", sh_path],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
        for line in iter(proc.stdout.readline, b""):
            logf.write(line)
            try:    print(line.decode("utf-8"),    end="")
            except Exception: print(line.decode("latin-1"), end="")
        proc.wait()
        code = proc.returncode

    print(f"\nLog: {log_path}")
    if code != 0:
        print(f"\nTraining failed; log tail:")
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            print("".join(f.readlines()[-200:]))
        raise SystemExit(code)

    adapters = _find_adapters(output_dir)
    if not adapters:
        raise RuntimeError("adapter_config.json was not found")

    print("\nFound LoRA adapter dirs:")
    for i, (step, _, d) in enumerate(adapters[:5]):
        print(f"  [{i}] {d} (step={step})")
    best_local = adapters[0][2]

    Path(adapter_dir).mkdir(parents=True, exist_ok=True)
    for fn in ["adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"]:
        src = os.path.join(best_local, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(adapter_dir, fn))
            print(f"  Copied {fn}")

    torch.cuda.empty_cache(); gc.collect()
    print(f"Training complete: {adapter_dir}")
    return {"base_model": base_model, "lora_dir": adapter_dir, "log_path": log_path}


# =========================================================================
# Implementation note.
# =========================================================================

# Implementation note.
def load_json_list(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return d["data"] if isinstance(d, dict) and "data" in d else d

def ensure_image(path: str) -> Optional[Image.Image]:
    try:
        return Image.open(path).convert("RGB")
    except (UnidentifiedImageError, FileNotFoundError, OSError, KeyError):
        return None

def build_messages_with_image(prompt_text: str) -> List[Dict[str, Any]]:
    return [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}]

def get_img_rel(ex: Dict[str, Any]) -> str:
    if "images" in ex:
        return ex["images"][0]
    return ex["image"]

def extract_assistant_text(rec: Dict[str, Any]) -> str:
    if "caption" in rec:
        return str(rec["caption"])
    msgs = rec.get("messages", [])
    if not msgs:
        return ""
    cont = msgs[-1].get("content", "")
    if isinstance(cont, str):
        return cont
    if isinstance(cont, list):
        out = [seg.get("text", "") for seg in cont
               if isinstance(seg, dict) and seg.get("type") == "text"]
        return " ".join(out).strip()
    if isinstance(cont, dict):
        return cont.get("text", "")
    return str(cont)


# Implementation note.
def load_model_with_lora(base_model_path: str, lora_dir: str):
    from transformers import AutoProcessor
    from transformers.models.qwen2_vl import Qwen2VLForConditionalGeneration
    from peft import PeftModel

    print("Loading base+LoRA ...")
    processor = AutoProcessor.from_pretrained(
        base_model_path, trust_remote_code=True, cache_dir=HF_CACHE_DIR)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        base_model_path, device_map="auto", torch_dtype=DTYPE,
        trust_remote_code=True, cache_dir=HF_CACHE_DIR)
    print("Using LoRA adapter:", lora_dir)
    model = PeftModel.from_pretrained(model, lora_dir, is_trainable=False)
    # Implementation note.
    import torch, os
    from safetensors import safe_open as _sopen
    import glob as _glob
    _md = dict(model.named_parameters())
    _af = _glob.glob(os.path.join(lora_dir, "adapter_model.safetensors"))
    if _af:
        _cnt = 0
        with _sopen(_af[0], framework="pt") as _sf:
            for _k in _sf.keys():
                if _k in _md:
                    with torch.no_grad():
                        _md[_k].copy_(_sf.get_tensor(_k).to(_md[_k].device, _md[_k].dtype))
                    _cnt += 1
        print(f">>> Injected LoRA weights: {_cnt} tensors <<<")
    # Implementation note.
    model.eval()
    return model, processor

def mark_trainable_lora_params(m) -> List[str]:
    sel = []
    for n, p in m.named_parameters():
        is_lora = ("lora" in n.lower()) and (p.ndim >= 2)
        p.requires_grad_(is_lora)
        if is_lora:
            sel.append(n)
    return sel


# Implementation note.
def _prefix_len_tokens(image: Image.Image, prompt_text: str, processor) -> int:
    msgs = build_messages_with_image(prompt_text)
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    batch = processor(images=[image], text=[text], return_tensors="pt")
    return int(batch["input_ids"].shape[1])

def _autocast_ctx():
    if DEVICE == "cuda":
        if DTYPE == torch.bfloat16:
            return torch.autocast("cuda", dtype=torch.bfloat16)
        elif DTYPE == torch.float16:
            return torch.autocast("cuda", dtype=torch.float16)
    return torch.autocast("cpu", enabled=False)

def backward_and_collect(img_path: str, target_text: str,
                         model, processor, vocab: int) -> Dict[str, torch.Tensor]:
    image = ensure_image(img_path)
    if image is None:
        return {}
    try:
        pl = _prefix_len_tokens(image, PROMPT_EN, processor)
        msgs_full = build_messages_with_image(PROMPT_EN) + [
            {"role": "assistant", "content": [{"type": "text", "text": target_text}]}
        ]
        text_full = processor.apply_chat_template(
            msgs_full, tokenize=False, add_generation_prompt=False)
        batch = processor(images=[image], text=[text_full],
                          return_tensors="pt").to(model.device)

        # Implementation note.
        # Implementation note.
        batch.pop("attention_mask", None)

        input_ids = batch["input_ids"]
        labels = input_ids.clone()
        labels[:, :pl] = -100

        # Implementation note.
        # Implementation note.
        # Implementation note.
        labels[(labels >= vocab) | (labels < -100)] = -100

        model.train()
        model.zero_grad(set_to_none=True)

        with _autocast_ctx():
            out = model(**batch, labels=labels)
            loss = out.loss
        if (loss is None) or (not torch.isfinite(loss)):
            model.zero_grad(set_to_none=True); model.eval()
            return {}
        loss.backward()

        grad_dict = {}
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            g = p.grad
            if g is None or g.ndim < 2:
                continue
            grad_dict[n] = g.detach().to(torch.float32).cpu()

        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache(); gc.collect()
        model.eval()
        return grad_dict
    except Exception:
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache(); gc.collect()
        model.eval()
        return {}

def row_col_cos(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if a.ndim < 2:
        s = F.cosine_similarity(a.flatten().unsqueeze(0), b.flatten().unsqueeze(0), dim=1)
        s = torch.nan_to_num(s, 0.0)
        return s, s
    r = F.cosine_similarity(a, b, dim=1)
    c = F.cosine_similarity(a.T, b.T, dim=1)
    return torch.nan_to_num(r, 0.0), torch.nan_to_num(c, 0.0)


# Implementation note.
def build_reference_grad(samples, root_dir, model, processor, vocab):
    ref = {}
    ok = 0
    for ex in tqdm(samples, desc="Reference gradients (members)"):
        img_path = os.path.join(root_dir, get_img_rel(ex))
        target = extract_assistant_text(ex)
        if not target:
            continue
        gd = backward_and_collect(img_path, target, model, processor, vocab)
        if not gd:
            continue
        ok += 1
        if not ref:
            ref = {k: v.clone() for k, v in gd.items()}
        else:
            for k in list(ref.keys()):
                if (k in gd) and (ref[k].shape == gd[k].shape):
                    ref[k] += gd[k].to(ref[k].dtype)
                else:
                    ref.pop(k, None)
    if ok == 0:
        raise RuntimeError("Reference-gradient construction failed: no calibration member produced gradients.")
    for k in ref:
        ref[k] /= ok
    print("  Valid reference samples:", ok, "reference-gradient keys:", len(ref))
    return ref

def avg_rowcol_sims(samples, ref, root_dir, tag, model, processor, vocab):
    acc_r, acc_c = {}, {}
    cnt = 0
    for ex in tqdm(samples, desc="Similarity (%s)" % tag):
        img_path = os.path.join(root_dir, get_img_rel(ex))
        target = extract_assistant_text(ex)
        if not target:
            continue
        gd = backward_and_collect(img_path, target, model, processor, vocab)
        if not gd:
            continue
        cnt += 1
        for name, g in gd.items():
            if (name not in ref) or (ref[name].shape != g.shape):
                continue
            rs, cs = row_col_cos(g, ref[name])
            if name not in acc_r:
                acc_r[name], acc_c[name] = rs.clone(), cs.clone()
            else:
                if acc_r[name].shape == rs.shape: acc_r[name] += rs
                if acc_c[name].shape == cs.shape: acc_c[name] += cs
    if cnt == 0:
        return {}, {}, 0
    for k in acc_r:
        acc_r[k] /= cnt; acc_c[k] /= cnt
    return acc_r, acc_c, cnt


# Implementation note.
def build_masks(mem_rc, non_rc, q=QUANTILE_Q, topk_fallback=TOP_K_FALLBACK):
    mr, mc = mem_rc; nr, nc = non_rc
    row_masks, col_masks = {}, {}
    total = 0
    for name in mr:
        if (name not in nr) or (name not in mc) or (name not in nc):
            continue
        rgap = torch.nan_to_num((mr[name] - nr[name]).detach().to(torch.float32).cpu(), 0.0)
        cgap = torch.nan_to_num((mc[name] - nc[name]).detach().to(torch.float32).cpu(), 0.0)
        rm = torch.zeros_like(rgap, dtype=torch.bool)
        cm = torch.zeros_like(cgap, dtype=torch.bool)
        if rgap.numel() > 0:
            rt = torch.quantile(rgap, q); rm = rgap > rt
        if cgap.numel() > 0:
            ct = torch.quantile(cgap, q); cm = cgap > ct
        if (not rm.any()) and (rgap.numel() > 0):
            k = min(topk_fallback, rgap.numel())
            rm[torch.topk(rgap, k=k).indices] = True
        if (not cm.any()) and (cgap.numel() > 0):
            k = min(topk_fallback, cgap.numel())
            cm[torch.topk(cgap, k=k).indices] = True
        if rm.any() or cm.any():
            row_masks[name] = rm; col_masks[name] = cm
            total += int(rm.sum().item() + cm.sum().item())
    return row_masks, col_masks, total


# Implementation note.
def gradsafe_score(ex, ref, row_masks, col_masks, root_dir, model, processor, vocab):
    img_path = os.path.join(root_dir, get_img_rel(ex))
    target = extract_assistant_text(ex)
    if not target:
        return 0.0
    gd = backward_and_collect(img_path, target, model, processor, vocab)
    if not gd:
        return 0.0
    sims = []
    for name, g in gd.items():
        if (name in ref) and (name in row_masks) and (name in col_masks)\
                and (ref[name].shape == g.shape):
            rs, cs = row_col_cos(g, ref[name])
            rm, cm = row_masks[name], col_masks[name]
            if rm.any(): sims.extend(rs[rm].cpu().tolist())
            if cm.any(): sims.extend(cs[cm].cpu().tolist())
    return float(np.mean(sims)) if sims else 0.0


# Implementation note.
def run_gradsafe_attack(model_info: Dict, out_csv: str, out_dir: str, config: Dict):
    print(f"\n{'='*70}")
    print("Section 3: GradSafe / GradAudit evaluation")
    print(f"{'='*70}")

    set_all_seeds(ATTACK_SEED)

    members    = load_json_list(MEMBER_JSON)
    nonmembers = load_json_list(NONMEMBER_JSON)
    assert len(members) >= 1000 and len(nonmembers) >= 1000, "The evaluation requires 1,000 members and 1,000 nonmembers"
    print("Eval ready: members=%d nonmembers=%d" % (len(members), len(nonmembers)))
    print("Device=%s dtype=%s" % (DEVICE, DTYPE))

    model, processor = load_model_with_lora(
        model_info["base_model"], model_info["lora_dir"])

    selected_params = mark_trainable_lora_params(model)
    print("Trainable gradient parameters:", len(selected_params), "(LoRA tensors)")
    if len(selected_params) == 0:
        raise RuntimeError("No LoRA tensors were found.")

    vocab = int(getattr(model.config, "vocab_size", 151936))

    # Implementation note.
    print("Quantile q=%s" % QUANTILE_Q)
    random.Random(ATTACK_SEED).shuffle(members)
    random.Random(ATTACK_SEED).shuffle(nonmembers)

    calib_members = members[:CALIB_MEM]
    calib_nonmems = nonmembers[:CALIB_NON]
    probe_members = members[CALIB_MEM:CALIB_MEM + PROBE_MEM]
    probe_nonmems = nonmembers[CALIB_NON:CALIB_NON + PROBE_NON]
    print("Split | calibration M=%d N=%d  probe M=%d N=%d" % (
        len(calib_members), len(calib_nonmems),
        len(probe_members), len(probe_nonmems)))

    # Implementation note.
    ref = build_reference_grad(
        calib_members, MEMBER_ROOT, model, processor, vocab)

    # Implementation note.
    mr, mc, m_cnt = avg_rowcol_sims(
        calib_members, ref, MEMBER_ROOT, "member", model, processor, vocab)
    nr, nc, n_cnt = avg_rowcol_sims(
        calib_nonmems, ref, NONMEMBER_ROOT, "nonmember", model, processor, vocab)
    print("Valid calibration samples: member=%d nonmember=%d" % (m_cnt, n_cnt))
    if (not mr) or (not nr):
        raise RuntimeError("Sensitive-mask construction failed: calibration similarities are empty.")

    # Implementation note.
    row_masks, col_masks, total_crit = build_masks(
        (mr, mc), (nr, nc), q=QUANTILE_Q, topk_fallback=TOP_K_FALLBACK)
    print("Total sensitive dimensions:", total_crit)

    # Implementation note.
    scores, labels = [], []
    probe_records = []
    for ex in tqdm(probe_members, desc="Probe Members"):
        s = gradsafe_score(ex, ref, row_masks, col_masks, MEMBER_ROOT, model, processor, vocab)
        scores.append(s); labels.append(1)
        probe_records.append({"image": get_img_rel(ex), "score": float(s),
                              "label": 1, "split": "probe_member"})
    for ex in tqdm(probe_nonmems, desc="Probe NonMembers"):
        s = gradsafe_score(ex, ref, row_masks, col_masks, NONMEMBER_ROOT, model, processor, vocab)
        scores.append(s); labels.append(0)
        probe_records.append({"image": get_img_rel(ex), "score": float(s),
                              "label": 0, "split": "probe_nonmember"})

    scores = np.array(scores); labels = np.array(labels)
    auc = roc_auc_score(labels, scores)
    fpr, tpr, thr = roc_curve(labels, scores)
    best_thr = float(thr[int(np.argmax(tpr - fpr))])
    pred = (scores >= best_thr).astype(int)
    acc = float(accuracy_score(labels, pred))
    cm  = confusion_matrix(labels, pred)

    print("\n====== GradSafe MIA results (Qwen2-VL + LoRA) ======")
    print("AUC  = %.4f" % auc)
    print("Threshold = %.4f" % best_thr)
    print("Acc  = %.4f" % acc)
    print("Confusion [[TN FP][FN TP]]:")
    print(cm)

    # Implementation note.
    import pandas as pd
    Path(os.path.dirname(out_csv)).mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"score": scores, "label": labels}).to_csv(
        out_csv, index=False, encoding="utf-8")
    print("Saved detailed CSV:", out_csv)

    # Implementation note.
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary = {
        "timestamp": ts,
        "method": "GradSafe/GradAudit (code-2 logic)",
        "config": {
            "model":          "Qwen2-VL-2B (MedTrinity)",
            "config_id":      config["id"],
            "attack_seed":    ATTACK_SEED,
            "quantile_q":     QUANTILE_Q,
            "top_k_fallback": TOP_K_FALLBACK,
            "calib_mem":      CALIB_MEM,
            "calib_non":      CALIB_NON,
            "probe_mem":      PROBE_MEM,
            "probe_non":      PROBE_NON,
            "prompt":         PROMPT_EN,
            "score_agg":      "mean",
            "mask_type":      "pure_quantile_top16_fallback",
            "ref_grad_source": "member_only",
            "lora_dir":       model_info.get("lora_dir", ""),
            "base_model":     model_info.get("base_model", ""),
        },
        "calib_valid_samples": {"member": int(m_cnt), "nonmember": int(n_cnt)},
        "total_sensitive_dims": int(total_crit),
        "metrics": {
            "auc":              float(auc),
            "best_threshold":   best_thr,
            "accuracy":         acc,
            "confusion_matrix": cm.tolist(),
        },
        "roc_data": {
            "fpr":        fpr.tolist(),
            "tpr":        tpr.tolist(),
            "thresholds": thr.tolist(),
        },
    }
    summary_path = os.path.join(out_dir, f"gradsafe_summary_{ts}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("Saved summary JSON:", summary_path)

    samples_path = os.path.join(out_dir, f"gradsafe_sample_scores_{ts}.json")
    with open(samples_path, "w", encoding="utf-8") as f:
        json.dump(probe_records, f, indent=2, ensure_ascii=False)
    print("Saved per-sample JSON:", samples_path)

    del model, processor
    torch.cuda.empty_cache(); gc.collect()


# =========================================================================
# Implementation note.
# =========================================================================
def main():
    args = parse_args()
    setup_train_env()

    config   = {**CONFIG, "id": args.config_id}
    work_dir = WORK_DIR
    Path(work_dir).mkdir(parents=True, exist_ok=True)

    # Implementation note.
    if args.fresh and not args.skip_train:
        clean_previous_outputs(config, work_dir)

    data_paths = prepare_training_data(config, work_dir)
    model_info = train_lora(
        config, data_paths["flat_train_json"],
        work_dir, args.base_model, args.llama_repo,
        skip_train=args.skip_train)

    # Implementation note.
    run_gradsafe_attack(model_info, args.out_csv, args.out_dir, config)

    print("\nExperiment complete!")


if __name__ == "__main__":
    main()
