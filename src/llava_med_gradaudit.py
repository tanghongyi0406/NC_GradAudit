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
#
# See README.md for implementation and reproducibility details.
# =========================================================================

import os, sys, json, gc, random, re, argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

import numpy as np
from PIL import Image, UnidentifiedImageError
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F

from sklearn.metrics import (
    roc_auc_score, roc_curve, accuracy_score, confusion_matrix,
    precision_score, recall_score, f1_score, precision_recall_fscore_support
)

# =========================================================================
# See README.md for implementation and reproducibility details.
# =========================================================================
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR      = str(PACKAGE_ROOT / "data" / "pmcoa_roco_mia")
DATASET_JSON = os.path.join(OUT_DIR, "dataset.json")
IMG_ROOT     = OUT_DIR

SEED = 42
NUM_MEMBERS = 1000
NUM_NONMEMBERS = 1000
CALIB_MEM = 200
CALIB_NON = 200
PROBE_MEM = 800
PROBE_NON = 800
SENSITIVITY_TAU = 0.10
VISION_LAST_N_BLOCKS = 3
USER_PROMPT = "<image>\nDescribe the medical image."
MODEL_ID = "Eren-Senoglu/llava-med-v1.5-mistral-7b-hf"
MODEL_REVISION = "f21ad87f3576d306fe97c4e2f77be1a859a84c7f"

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", type=str,
                   default=str(PACKAGE_ROOT / "results" / "llava_med"))
    p.add_argument("--dataset_json", type=str, default=DATASET_JSON)
    p.add_argument("--img_root", type=str, default=IMG_ROOT)
    p.add_argument("--model_id", type=str, default=MODEL_ID)
    return p.parse_args()

# =========================================================================
# See README.md for implementation and reproducibility details.
# =========================================================================
def setup_env():
    assert torch.cuda.is_available(), " GPU"
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    global DTYPE
    DTYPE = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
    print(f"✅ Device: cuda:0  GPU: {torch.cuda.get_device_name(0)}  dtype={DTYPE}")

DTYPE = torch.float16

# =========================================================================
# See README.md for implementation and reproducibility details.
# =========================================================================
def load_model(model_id=MODEL_ID):
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    print(f"\n⏳ Loading model: {model_id}")
    revision = MODEL_REVISION if model_id == MODEL_ID else None
    processor = AutoProcessor.from_pretrained(
        model_id, revision=revision, trust_remote_code=True)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, revision=revision, dtype=DTYPE,
        device_map=None, trust_remote_code=True).to("cuda:0")
    try:
        model.gradient_checkpointing_disable()
    except Exception:
        pass
    if hasattr(model, "config"):
        model.config.use_cache = False
    model.eval()
    if getattr(processor, "patch_size", None) is None:
        ip = getattr(processor, "image_processor", None)
        ps = getattr(ip, "patch_size", None)
        processor.patch_size = ps if isinstance(ps, int) else 14
    if getattr(processor, "num_additional_image_tokens", None) is None:
        processor.num_additional_image_tokens = 0
    vt = getattr(model, "vision_tower", None)
    if vt is not None:
        try: vt.to("cuda:0")
        except Exception:
            inner = getattr(vt, "vision_tower", None)
            if inner is not None: inner.to("cuda:0")
    print("✅ ")
    return model, processor

# =========================================================================
# See README.md for implementation and reproducibility details.
# =========================================================================
def load_and_split(dataset_json: str):
    with open(dataset_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    members_all = [x for x in data if x["label"] == 1]
    nonmembers_all = [x for x in data if x["label"] == 0]
    split_path = PACKAGE_ROOT / "configs" / "llava_med_split.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    calib_member_ids = set(split["calibration_member_images"])
    calib_nonmember_ids = set(split["calibration_nonmember_images"])
    print(f"\n📦 members={len(members_all)} | non-members={len(nonmembers_all)}")
    calib_members = [x for x in members_all if x["image"] in calib_member_ids]
    calib_nonmems = [x for x in nonmembers_all if x["image"] in calib_nonmember_ids]
    probe_members = [x for x in members_all if x["image"] not in calib_member_ids]
    probe_nonmems = [x for x in nonmembers_all if x["image"] not in calib_nonmember_ids]
    counts = tuple(map(len, (calib_members, calib_nonmems, probe_members, probe_nonmems)))
    if counts != (CALIB_MEM, CALIB_NON, PROBE_MEM, PROBE_NON):
        raise RuntimeError(f"Frozen LLaVA-Med split mismatch: {counts}")
    print(f"Calibration: M={len(calib_members)}, N={len(calib_nonmems)} | "
          f"Probe: M={len(probe_members)}, N={len(probe_nonmems)}")
    return calib_members, calib_nonmems, probe_members, probe_nonmems

def load_image(sample, img_root):
    rel = sample.get("image", "")
    if not rel: return None
    p = os.path.join(img_root, rel)
    if not os.path.exists(p): return None
    try:
        return Image.open(p).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError, Exception):
        return None

# =========================================================================
# 4) prompt cache
# =========================================================================
USER_PROMPT_STR = ""
PREFIX_LEN = 0

def init_prompt_cache(processor):
    global USER_PROMPT_STR, PREFIX_LEN
    USER_PROMPT_STR = "<s> [INST] <image>\nDescribe the medical image. [/INST]"
    prefix_ids = processor.tokenizer(USER_PROMPT_STR, return_tensors="pt").input_ids
    PREFIX_LEN = int(prefix_ids.size(1))
    print(f"🔧 prefix_len={PREFIX_LEN}")

def build_batch(processor, image, caption):
    full_text = USER_PROMPT_STR + str(caption)
    enc = processor(text=full_text, images=[image], return_tensors="pt")
    labels = enc["input_ids"].clone()
    labels[:, :PREFIX_LEN] = -100
    enc = {k: (v.to("cuda:0", non_blocking=True) if torch.is_tensor(v) else v)
           for k, v in enc.items()}
    enc["labels"] = labels.to("cuda:0", non_blocking=True)
    return enc

# =========================================================================
# See README.md for implementation and reproducibility details.
# =========================================================================
def enable_grads_selectively(model):
    for _, p in model.named_parameters():
        p.requires_grad_(False)
    names = []
    for name, p in model.named_parameters():
        if re.search(r"(multi_modal_projector|mm_projector|vision_proj|projector)", name):
            if p.ndim >= 2:
                p.requires_grad_(True); names.append(name)
            continue
        if "vision_tower" in name and ("encoder.layers" in name or "vision_model.encoder.layers" in name):
            mobj = re.search(r"encoder\.layers\.(\d+)\.", name)
            if mobj and p.ndim >= 2 and re.search(
                    r"(q_proj|k_proj|v_proj|o_proj|in_proj|fc|mlp|proj|dense|linear|qkv)", name):
                names.append(name); p.requires_grad_(True)
            continue
        if re.search(r"(language_model|model\.layers\.)", name):
            if re.search(r"(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))", name):
                if p.ndim >= 2:
                    p.requires_grad_(True); names.append(name)
    return names

# =========================================================================
# See README.md for implementation and reproducibility details.
# =========================================================================
def row_col_cosine(g: torch.Tensor, ref: torch.Tensor):
    """GPU-resident row/column cosine with bounded FP32 temporaries."""
    target_elements = 2_000_000
    if g.ndim < 2:
        flat = F.cosine_similarity(g.flatten().float().unsqueeze(0),
                                   ref.flatten().float().unsqueeze(0), dim=1)
        flat = torch.nan_to_num(flat, nan=0.0)
        return flat, flat
    row_step = max(1, target_elements // max(g.shape[1], 1))
    col_step = max(1, target_elements // max(g.shape[0], 1))
    rows = [F.cosine_similarity(g[i:i+row_step].float(), ref[i:i+row_step].float(), dim=1)
            for i in range(0, g.shape[0], row_step)]
    cols = [F.cosine_similarity(g[:, i:i+col_step].float(), ref[:, i:i+col_step].float(), dim=0)
            for i in range(0, g.shape[1], col_step)]
    return torch.nan_to_num(torch.cat(rows), nan=0.0), torch.nan_to_num(torch.cat(cols), nan=0.0)

# =========================================================================
# See README.md for implementation and reproducibility details.
# =========================================================================
def forward_backward(model, batch):
    """Run forward/backward and retain parameter gradients on the GPU."""
    model.train()
    model.zero_grad(set_to_none=True)
    with torch.amp.autocast('cuda', dtype=DTYPE):
        out = model(**batch)
        loss = out.loss
    if loss is None or not torch.isfinite(loss):
        model.zero_grad(set_to_none=True)
        return False
    loss.backward()
    del out, loss
    return True

def cleanup(model):
    model.zero_grad(set_to_none=True)
    model.eval()

# =========================================================================
# See README.md for implementation and reproducibility details.
# =========================================================================
def average_reference_grad(model, processor, members, img_root, take):
    ref: Dict[str, torch.Tensor] = {}   # See README.md for implementation and reproducibility details.
    ok = 0
    for i in tqdm(range(take), desc="()"):
        obj = members[i]
        img = load_image(obj, img_root)
        if img is None:
            continue
        cap = obj.get("caption", "")
        try:
            batch = build_batch(processor, img, cap)
            if not forward_backward(model, batch):
                cleanup(model); continue
        except Exception as e:
            import traceback
            if i < 3:
                print(f"\n[{i}] : {e}"); traceback.print_exc()
            cleanup(model); continue

        # See README.md for implementation and reproducibility details.
        ok += 1
        for name, p in model.named_parameters():
            if p.requires_grad and (p.grad is not None) and (p.grad.ndim >= 2):
                g_cpu = p.grad.detach().to(torch.float16)
                if name not in ref:
                    ref[name] = g_cpu.clone()
                else:
                    if ref[name].shape == g_cpu.shape:
                        ref[name].add_(g_cpu)
                    else:
                        ref.pop(name, None)
        cleanup(model)   # See README.md for implementation and reproducibility details.

    if ok == 0:
        raise RuntimeError("Reference-gradient construction produced no valid samples.")
    for k in ref:
        ref[k] = ref[k].div_(ok)
    print(f"  ✅ {ok}/{take} , {len(ref)} ")
    return ref

# =========================================================================
# See README.md for implementation and reproducibility details.
# =========================================================================
def avg_rowcol_similarity(model, processor, samples, ref_grads, img_root, is_member, take):
    acc_row: Dict[str, torch.Tensor] = {}
    acc_col: Dict[str, torch.Tensor] = {}
    count = 0
    label = "M" if is_member else "N"
    for i in tqdm(range(take), desc=f"({label})"):
        obj = samples[i]
        img = load_image(obj, img_root)
        if img is None:
            continue
        cap = obj.get("caption", "")
        try:
            batch = build_batch(processor, img, cap)
            if not forward_backward(model, batch):
                cleanup(model); continue
        except Exception as e:
            import traceback
            if i < 3:
                print(f"\n[{i}] : {e}"); traceback.print_exc()
            cleanup(model); continue

        count += 1
        # See README.md for implementation and reproducibility details.
        for name, p in model.named_parameters():
            if not (p.requires_grad and p.grad is not None and p.grad.ndim >= 2):
                continue
            if name not in ref_grads or ref_grads[name].shape != p.grad.shape:
                continue
            rs, cs = row_col_cosine(p.grad.detach(), ref_grads[name])
            if name not in acc_row:
                acc_row[name] = rs.clone()
                acc_col[name] = cs.clone()
            else:
                if acc_row[name].shape == rs.shape: acc_row[name].add_(rs)
                if acc_col[name].shape == cs.shape: acc_col[name].add_(cs)
        cleanup(model)

    if count == 0:
        print(f"  ⚠️ {label} ")
        return {}, {}
    for k in list(acc_row.keys()):
        acc_row[k].div_(count); acc_col[k].div_(count)
    print(f"  ✅ {count}/{take}")
    return acc_row, acc_col

def build_critical_masks(member_rc, nonmember_rc, tau=SENSITIVITY_TAU):
    mr, mc = member_rc
    nr, nc = nonmember_rc
    row_masks, col_masks, total = {}, {}, 0
    if not mr or not mc:
        return {}, {}, 0
    for name in mr:
        if name not in nr or name not in mc or name not in nc:
            continue
        try:
            rgap = torch.nan_to_num((mr[name] - nr[name]).to(torch.float32), nan=0.0)
            cgap = torch.nan_to_num((mc[name] - nc[name]).to(torch.float32), nan=0.0)
            row_masks[name] = (rgap > tau)
            col_masks[name] = (cgap > tau)
            total += int(row_masks[name].sum().item() + col_masks[name].sum().item())
        except Exception:
            continue
    return row_masks, col_masks, total

# =========================================================================
# See README.md for implementation and reproducibility details.
# =========================================================================
def gradsafe_score_for_item(model, processor, obj, img_root, ref_grads, row_masks, col_masks):
    img = load_image(obj, img_root)
    if img is None:
        return 0.0
    cap = obj.get("caption", "")
    try:
        batch = build_batch(processor, img, cap)
        if not forward_backward(model, batch):
            cleanup(model); return 0.0
    except Exception:
        cleanup(model); return 0.0

    total = torch.zeros((), device="cuda:0", dtype=torch.float32)
    count = 0
    for name, p in model.named_parameters():
        if not (p.requires_grad and p.grad is not None and p.grad.ndim >= 2):
            continue
        if name in ref_grads and name in row_masks and name in col_masks:
            if ref_grads[name].shape != p.grad.shape:
                continue
            rs, cs = row_col_cosine(p.grad.detach(), ref_grads[name])
            rm, cm = row_masks[name], col_masks[name]
            if rm.any():
                selected = rs[rm]; total.add_(selected.sum()); count += selected.numel()
            if cm.any():
                selected = cs[cm]; total.add_(selected.sum()); count += selected.numel()
    score = (total / max(count, 1)).item()
    cleanup(model)
    return float(score) if count else 0.0

def probe_set_score(model, processor, probe_members, probe_nonmems, img_root,
                    ref_grads, row_masks, col_masks):
    scores, labels, records = [], [], []
    for obj in tqdm(probe_members, desc="Probe-Members"):
        s = gradsafe_score_for_item(model, processor, obj, img_root, ref_grads, row_masks, col_masks)
        scores.append(s); labels.append(1)
        records.append({"image": str(obj.get("image","?")),
                        "caption": str(obj.get("caption",""))[:120],
                        "score": float(s), "label": 1, "split": "probe_member"})
    for obj in tqdm(probe_nonmems, desc="Probe-NonMembers"):
        s = gradsafe_score_for_item(model, processor, obj, img_root, ref_grads, row_masks, col_masks)
        scores.append(s); labels.append(0)
        records.append({"image": str(obj.get("image","?")),
                        "caption": str(obj.get("caption",""))[:120],
                        "score": float(s), "label": 0, "split": "probe_nonmember"})
    return np.array(scores), np.array(labels), records

# =========================================================================
# See README.md for implementation and reproducibility details.
# =========================================================================
def compute_metrics(scores, labels):
    if np.all(scores == scores[0]):
        return {"auc": 0.5, "best_threshold": float(np.median(scores)), "accuracy": 0.5,
                "confusion_matrix": None, "fpr5_metrics": None,
                "roc_data": {"fpr": [], "tpr": [], "thresholds": []}}
    auc = roc_auc_score(labels, scores)
    fpr_arr, tpr_arr, thr_arr = roc_curve(labels, scores)
    j = int(np.argmax(tpr_arr - fpr_arr))
    best_thr = float(thr_arr[j])
    pred = (scores >= best_thr).astype(int)
    acc = accuracy_score(labels, pred)
    cm = confusion_matrix(labels, pred).tolist()
    nonmask = (labels == 0)
    fpr5 = None
    if nonmask.sum() > 0:
        thr_5 = float(np.quantile(scores[nonmask], 0.95))
        pred_5 = (scores >= thr_5).astype(int)
        cm_5 = confusion_matrix(labels, pred_5)
        tn, fp, fn, tp = cm_5.ravel()
        prec_5, rec_5, f1_5, _ = precision_recall_fscore_support(
            labels, pred_5, average="binary", zero_division=0)
        fpr5 = {"threshold": thr_5, "actual_fpr": float(fp/(fp+tn+1e-12)),
                "tpr": float(tp/(tp+fn+1e-12)), "precision": float(prec_5),
                "recall": float(rec_5), "f1": float(f1_5),
                "accuracy": float(accuracy_score(labels, pred_5)),
                "confusion_matrix": cm_5.tolist()}
    return {"auc": float(auc), "best_threshold": float(best_thr), "accuracy": float(acc),
            "confusion_matrix": cm, "fpr5_metrics": fpr5,
            "roc_data": {"fpr": fpr_arr.tolist(), "tpr": tpr_arr.tolist(), "thresholds": thr_arr.tolist()}}

def save_results(output_dir, metrics, sample_records):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = {"timestamp": ts,
               "config": {"model": MODEL_ID, "seed": SEED, "calib_mem": CALIB_MEM,
                          "calib_non": CALIB_NON, "probe_mem": PROBE_MEM, "probe_non": PROBE_NON,
                          "sensitivity_tau": SENSITIVITY_TAU, "score_agg": "mean",
                          "mask_type": "fixed_tau", "ref_grad_source": "member_only",
                          "grad_mode": "streaming_no_store"},
               "metrics": {k: v for k, v in metrics.items() if k != "roc_data"}}
    p1 = os.path.join(output_dir, f"results_summary_{ts}.json")
    with open(p1, "w", encoding="utf-8") as f: json.dump(summary, f, indent=2, ensure_ascii=False)
    p2 = os.path.join(output_dir, f"sample_scores_{ts}.json")
    with open(p2, "w", encoding="utf-8") as f: json.dump(sample_records, f, indent=2, ensure_ascii=False)
    p3 = os.path.join(output_dir, f"roc_data_{ts}.json")
    with open(p3, "w", encoding="utf-8") as f: json.dump(metrics.get("roc_data", {}), f, indent=2, ensure_ascii=False)
    print(f"\n💾 : {p1}\n           {p2}\n           {p3}")

def print_metrics(metrics, total_crit):
    m = metrics
    print("\n====== GradSafe-only MIA  (LLaVA-Med, A) ======")
    print(f" (tau={SENSITIVITY_TAU}): {total_crit}")
    print(f"AUC  = {m.get('auc')}")
    print(f" = {m.get('best_threshold')}")
    print(f"Acc  = {m.get('accuracy')}")
    print("Confusion [[TN FP][FN TP]]:")
    print(np.array(m.get("confusion_matrix", [[0,0],[0,0]])))
    fp5 = m.get("fpr5_metrics")
    print("\n------ @5% FPR ------")
    if fp5:
        print(f"Threshold={fp5['threshold']:.6f} FPR={fp5['actual_fpr']*100:.2f}% "
              f"TPR={fp5['tpr']*100:.2f}% Prec={fp5['precision']*100:.2f}% F1={fp5['f1']:.4f}")
    else:
        print("⚠️ ")

# =========================================================================
# See README.md for implementation and reproducibility details.
# =========================================================================
def main():
    args = parse_args()
    setup_env()
    model, processor = load_model(args.model_id)
    init_prompt_cache(processor)
    calib_members, calib_nonmems, probe_members, probe_nonmems = load_and_split(args.dataset_json)
    collect_names = enable_grads_selectively(model)
    print(f"🔧 : {len(collect_names)}")
    if len(collect_names) == 0:
        print("❌ "); sys.exit(1)
    IR = args.img_root
    try:
        print("\n[Step 1/4] ()…")
        ref_grads = average_reference_grad(model, processor, calib_members, IR, len(calib_members))
        print("\n[Step 2/4] …")
        mr, mc = avg_rowcol_similarity(model, processor, calib_members, ref_grads, IR, True, len(calib_members))
        nr, nc = avg_rowcol_similarity(model, processor, calib_nonmems, ref_grads, IR, False, len(calib_nonmems))
        row_masks, col_masks, total_crit = build_critical_masks((mr, mc), (nr, nc), SENSITIVITY_TAU)
        print(f"  ✅ : {total_crit}")
        if total_crit == 0:
            print("⚠️ 0"); return
        print("\n[Step 3/4] …")
        scores, labels, sample_records = probe_set_score(
            model, processor, probe_members, probe_nonmems, IR, ref_grads, row_masks, col_masks)
        print("\n[Step 4/4]  & …")
        metrics = compute_metrics(scores, labels)
        print_metrics(metrics, total_crit)
        save_results(args.output_dir, metrics, sample_records)
    except Exception as e:
        import traceback
        print(f"\n❌ : {e}"); traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
