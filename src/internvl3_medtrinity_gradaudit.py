from __future__ import annotations

import csv
import gc
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


_IMAGE_TOKEN_RE = re.compile(r"<\s*image\s*>", flags=re.IGNORECASE)


def _lazy_import_runtime():
    import torch
    import torch.nn.functional as F
    from PIL import Image, UnidentifiedImageError
    from peft import PeftModel
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        precision_score,
        recall_score,
        roc_auc_score,
        roc_curve,
    )
    from transformers import AutoModelForImageTextToText, AutoProcessor

    return {
        "torch": torch,
        "F": F,
        "Image": Image,
        "UnidentifiedImageError": UnidentifiedImageError,
        "PeftModel": PeftModel,
        "accuracy_score": accuracy_score,
        "confusion_matrix": confusion_matrix,
        "precision_score": precision_score,
        "recall_score": recall_score,
        "roc_auc_score": roc_auc_score,
        "roc_curve": roc_curve,
        "AutoProcessor": AutoProcessor,
        "AutoModelForImageTextToText": AutoModelForImageTextToText,
    }


@dataclass
class LayerStrategy:
    name: str
    mode: str


@dataclass
class ExperimentSpec:
    config_id: str
    dataset_name: str
    train_mode: str
    member_json: str
    nonmember_json: str
    member_media_root: str
    nonmember_media_root: str
    train_prompt: str
    attack_prompt: str
    output_subdir: str
    train_size: int = 1000
    train_seed: int = 42
    split_seed_offset: int = 123
    train_epochs: float = 5.0
    learning_rate: float = 1e-4
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.0
    max_samples: int = 1000
    save_total_limit: int = 30
    cutoff_len: int = 2048
    ref_size: int = 200
    calib_member: int = 200
    calib_nonmember: int = 200
    probe_member: int = 800
    probe_nonmember: int = 800
    attack_seed: int = 42
    loss_mode: str = "assistant_only"
    mask_mode: str = "quantile_topk"
    sensitivity_tau: float = 0.10
    quantile_q: float = 0.80
    topk_fallback: int = 16
    vision_last_n: int = 3
    text_last_m: int = 3
    layer_strategies: List[LayerStrategy] = field(default_factory=list)


@dataclass
class RunArgs:
    base_dir: str
    llama_repo: str
    base_model: str
    hf_cache_dir: str
    work_dir: str
    adapter_store: str
    results_dir: str
    fresh: bool = False
    skip_train: bool = False
    trust_remote_code: bool = True
    verbose_grad_errors: bool = False


def strip_image_tokens(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    text = _IMAGE_TOKEN_RE.sub("", text).replace("\n", " ")
    return re.sub(r"\s+", " ", text).strip()


def ensure_dir(path: str) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def load_json_list(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    if not isinstance(data, list):
        raise ValueError(f"JSON list expected: {path}")
    return data


def extract_assistant_text(record: Dict[str, Any]) -> str:
    if "caption" in record:
        return str(record["caption"])
    messages = record.get("messages", [])
    if not messages:
        return ""
    content = messages[-1].get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return str(content.get("text", ""))
    if isinstance(content, list):
        parts = []
        for seg in content:
            if isinstance(seg, dict) and seg.get("type") == "text":
                parts.append(str(seg.get("text", "")))
            elif isinstance(seg, str):
                parts.append(seg)
        return " ".join(parts).strip()
    return str(content)


def extract_user_prompt(record: Dict[str, Any], default_prompt: str) -> str:
    messages = record.get("messages", [])
    if not messages:
        return default_prompt
    content = messages[0].get("content", "")
    if isinstance(content, str):
        text = strip_image_tokens(content)
        return text or default_prompt
    if isinstance(content, dict):
        text = strip_image_tokens(content.get("text", ""))
        return text or default_prompt
    if isinstance(content, list):
        parts = []
        for seg in content:
            if isinstance(seg, dict) and seg.get("type") == "text":
                parts.append(str(seg.get("text", "")))
            elif isinstance(seg, str):
                parts.append(seg)
        text = strip_image_tokens(" ".join(parts))
        return text or default_prompt
    return default_prompt


def normalize_record(record: Dict[str, Any], default_prompt: str) -> Optional[Dict[str, Any]]:
    images = record.get("images")
    if images is None and record.get("image"):
        images = [record["image"]]
    if not images:
        return None

    assistant_text = extract_assistant_text(record)
    if not assistant_text:
        return None

    # LLaMA-Factory's InternVL multimodal plugin requires one explicit image
    # token for every entry in the images column.
    user_prompt = f"<image>\n{extract_user_prompt(record, default_prompt)}"
    return {
        "images": [images[0]],
        "messages": [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_text},
        ],
    }


def build_training_and_attack_sets(
    spec: ExperimentSpec, work_dir: str
) -> Dict[str, str]:
    exp_dir = Path(work_dir) / spec.config_id
    dataset_dir = exp_dir / "lf_dataset"
    attack_dir = exp_dir / "attack_data"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    attack_dir.mkdir(parents=True, exist_ok=True)

    if spec.train_mode == "member_5k_split":
        members = load_json_list(spec.member_json)
        nonmembers = load_json_list(spec.nonmember_json)
        split_seed = spec.train_seed + spec.split_seed_offset
        random.Random(split_seed).shuffle(members)
        random.Random(split_seed).shuffle(nonmembers)

        train_records = []
        for item in members[: spec.train_size]:
            normalized = normalize_record(item, spec.train_prompt)
            if normalized is not None:
                train_records.append(normalized)

        member_eval = random.Random(split_seed).sample(
            train_records, min(1000, len(train_records))
        )
        nonmember_eval = []
        for item in nonmembers[:1000]:
            normalized = normalize_record(item, spec.train_prompt)
            if normalized is not None:
                nonmember_eval.append(normalized)

        fingerprint = {
            "split_seed": split_seed,
            "train_size": len(train_records),
            "member_eval_size": len(member_eval),
            "nonmember_eval_size": len(nonmember_eval),
            "member_0": member_eval[0]["images"][0] if member_eval else None,
            "nonmember_0": nonmember_eval[0]["images"][0] if nonmember_eval else None,
        }
    elif spec.train_mode == "member_eval_flat":
        members_raw = load_json_list(spec.member_json)
        nonmembers_raw = load_json_list(spec.nonmember_json)

        train_records = []
        member_eval = []
        nonmember_eval = []
        for item in members_raw:
            normalized = normalize_record(item, spec.train_prompt)
            if normalized is not None:
                train_records.append(normalized)
                member_eval.append(normalized)
        for item in nonmembers_raw:
            normalized = normalize_record(item, spec.train_prompt)
            if normalized is not None:
                nonmember_eval.append(normalized)

        fingerprint = {
            "train_size": len(train_records),
            "member_eval_size": len(member_eval),
            "nonmember_eval_size": len(nonmember_eval),
            "member_0": member_eval[0]["images"][0] if member_eval else None,
            "nonmember_0": nonmember_eval[0]["images"][0] if nonmember_eval else None,
        }
    else:
        raise ValueError(f"Unknown train_mode: {spec.train_mode}")

    train_json = dataset_dir / f"{spec.config_id}_train.json"
    member_eval_json = attack_dir / "member_eval.json"
    nonmember_eval_json = attack_dir / "nonmember_eval.json"
    dataset_info_json = dataset_dir / "dataset_info.json"
    fingerprint_json = attack_dir / "split_fingerprint.json"

    with open(train_json, "w", encoding="utf-8") as f:
        json.dump(train_records, f, ensure_ascii=False, indent=2)
    with open(member_eval_json, "w", encoding="utf-8") as f:
        json.dump(member_eval, f, ensure_ascii=False, indent=2)
    with open(nonmember_eval_json, "w", encoding="utf-8") as f:
        json.dump(nonmember_eval, f, ensure_ascii=False, indent=2)
    with open(fingerprint_json, "w", encoding="utf-8") as f:
        json.dump(fingerprint, f, ensure_ascii=False, indent=2)

    dataset_key = spec.config_id
    dataset_info = {
        dataset_key: {
            "file_name": train_json.name,
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
            },
        }
    }
    with open(dataset_info_json, "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)

    return {
        "exp_dir": str(exp_dir),
        "dataset_dir": str(dataset_dir),
        "train_json": str(train_json),
        "dataset_key": dataset_key,
        "member_eval_json": str(member_eval_json),
        "nonmember_eval_json": str(nonmember_eval_json),
        "fingerprint_json": str(fingerprint_json),
    }


def latest_checkpoint(root: str) -> Optional[str]:
    path = Path(root)
    if not path.exists():
        return None
    cands: List[Tuple[int, float, str]] = []
    for item in path.glob("checkpoint-*"):
        match = re.search(r"checkpoint-(\d+)$", item.name)
        if match:
            cands.append((int(match.group(1)), item.stat().st_mtime, str(item)))
    if not cands:
        return None
    cands.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return cands[0][2]


def find_adapter_dirs(root: str) -> List[Tuple[int, float, str]]:
    path = Path(root)
    hits: List[Tuple[int, float, str]] = []
    for item in path.rglob("adapter_config.json"):
        step = -1
        match = re.search(r"checkpoint[-_]?(\d+)", str(item.parent))
        if match:
            step = int(match.group(1))
        hits.append((step, item.parent.stat().st_mtime, str(item.parent)))
    if not hits and (path / "adapter_config.json").is_file():
        hits.append((10**9, path.stat().st_mtime, str(path)))
    hits.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return hits


def clean_previous_outputs(exp_dir: str, adapter_dir: str) -> None:
    for path in [exp_dir, adapter_dir]:
        if os.path.isdir(path):
            shutil.rmtree(path)


def train_with_llamafactory(
    spec: ExperimentSpec,
    args: RunArgs,
    data_paths: Dict[str, str],
) -> Dict[str, str]:
    runtime = _lazy_import_runtime()
    torch = runtime["torch"]

    if not torch.cuda.is_available():
        raise RuntimeError("LoRA training requires CUDA.")

    exp_dir = Path(data_paths["exp_dir"])
    train_out_dir = exp_dir / "lora_model"
    adapter_dir = Path(args.adapter_store) / spec.config_id
    log_path = exp_dir / "train.log"

    if args.fresh and not args.skip_train:
        clean_previous_outputs(str(train_out_dir), str(adapter_dir))
        exp_dir.mkdir(parents=True, exist_ok=True)

    adapter_ready = (
        (adapter_dir / "adapter_config.json").is_file()
        and (
            (adapter_dir / "adapter_model.safetensors").is_file()
            or (adapter_dir / "adapter_model.bin").is_file()
        )
    )
    if args.skip_train or adapter_ready:
        return {
            "base_model": args.base_model,
            "lora_dir": str(adapter_dir),
            "train_log": str(log_path),
        }

    ensure_dir(args.adapter_store)
    ensure_dir(str(train_out_dir))

    resume_dir = latest_checkpoint(str(train_out_dir))
    precision_arg = "--bf16" if torch.cuda.is_bf16_supported() else "--fp16"
    cmd = [
        "llamafactory-cli",
        "train",
        "--stage",
        "sft",
        "--do_train",
        "True",
        "--model_name_or_path",
        args.base_model,
        "--cache_dir",
        args.hf_cache_dir,
        "--finetuning_type",
        "lora",
        "--template",
        "intern_vl",
        "--media_dir",
        spec.member_media_root,
        "--dataset_dir",
        data_paths["dataset_dir"],
        "--dataset",
        data_paths["dataset_key"],
        "--cutoff_len",
        str(spec.cutoff_len),
        "--learning_rate",
        str(spec.learning_rate),
        "--num_train_epochs",
        str(spec.train_epochs),
        "--max_samples",
        str(spec.max_samples),
        "--per_device_train_batch_size",
        str(spec.per_device_train_batch_size),
        "--gradient_accumulation_steps",
        str(spec.gradient_accumulation_steps),
        "--lr_scheduler_type",
        "cosine",
        "--max_grad_norm",
        "1.0",
        "--logging_steps",
        "5",
        "--save_strategy",
        "steps",
        "--save_steps",
        "100",
        "--save_total_limit",
        str(spec.save_total_limit),
        "--warmup_steps",
        "0",
        "--optim",
        "adamw_torch",
        "--report_to",
        "none",
        "--output_dir",
        str(train_out_dir),
        "--overwrite_output_dir",
        "True",
        precision_arg,
        "True",
        "--plot_loss",
        "False",
        "--flash_attn",
        "sdpa",
        "--dataloader_num_workers",
        "0",
        "--no_dataloader_pin_memory",
        "--torch_empty_cache_steps",
        "20",
        "--lora_rank",
        str(spec.lora_rank),
        "--lora_alpha",
        str(spec.lora_alpha),
        "--lora_dropout",
        str(spec.lora_dropout),
        "--lora_target",
        "all",
        "--gradient_checkpointing",
        "True",
    ]
    if args.trust_remote_code:
        cmd.extend(["--trust_remote_code", "True"])
    if resume_dir:
        cmd.extend(["--resume_from_checkpoint", resume_dir])

    env = os.environ.copy()
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    llama_src_dir = os.path.join(args.llama_repo, "src")
    prev_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{llama_src_dir}:{prev_pythonpath}" if prev_pythonpath else llama_src_dir
    )

    with open(log_path, "wb") as logf:
        proc = subprocess.Popen(
            cmd,
            cwd=args.llama_repo,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
        )
        assert proc.stdout is not None
        for line in iter(proc.stdout.readline, b""):
            logf.write(line)
            try:
                print(line.decode("utf-8"), end="")
            except Exception:
                print(line.decode("latin-1"), end="")
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"LLaMA-Factory training failed: {log_path}")

    adapters = find_adapter_dirs(str(train_out_dir))
    if not adapters:
        if log_path.exists():
            print("\n[train.log tail]")
            try:
                lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
                for line in lines[-200:]:
                    print(line)
            except Exception as exc:
                print(f"failed to read log tail: {exc}")
        if train_out_dir.exists():
            print("\n[lora_model files]")
            try:
                for item in sorted(train_out_dir.rglob("*"))[:200]:
                    print(item)
            except Exception as exc:
                print(f"failed to list output dir: {exc}")
        raise RuntimeError(f"No adapter_config.json found under {train_out_dir}")

    best_dir = Path(adapters[0][2])
    adapter_dir.mkdir(parents=True, exist_ok=True)
    for name in ["adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"]:
        src = best_dir / name
        if src.exists():
            shutil.copy2(src, adapter_dir / name)

    torch.cuda.empty_cache()
    gc.collect()
    return {
        "base_model": args.base_model,
        "lora_dir": str(adapter_dir),
        "train_log": str(log_path),
    }


def seed_everything(seed: int) -> None:
    runtime = _lazy_import_runtime()
    torch = runtime["torch"]

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_image(path: str):
    runtime = _lazy_import_runtime()
    Image = runtime["Image"]
    UnidentifiedImageError = runtime["UnidentifiedImageError"]
    try:
        return Image.open(path).convert("RGB")
    except (FileNotFoundError, OSError, KeyError, UnidentifiedImageError):
        return None


def patch_adapter_weights_if_available(model, lora_dir: str) -> None:
    safetensors_path = Path(lora_dir) / "adapter_model.safetensors"
    if not safetensors_path.is_file():
        return
    try:
        from safetensors import safe_open
    except Exception:
        return

    model_params = dict(model.named_parameters())
    copied = 0
    with safe_open(str(safetensors_path), framework="pt") as sf:
        for key in sf.keys():
            if key not in model_params:
                continue
            tensor = sf.get_tensor(key).to(
                model_params[key].device, model_params[key].dtype
            )
            model_params[key].data.copy_(tensor)
            copied += 1
    if copied:
        print(f"Patched {copied} LoRA tensors from {safetensors_path.name}")


def load_model_with_lora(base_model_path: str, lora_dir: str, hf_cache_dir: str):
    runtime = _lazy_import_runtime()
    torch = runtime["torch"]
    AutoProcessor = runtime["AutoProcessor"]
    AutoModelForImageTextToText = runtime["AutoModelForImageTextToText"]
    PeftModel = runtime["PeftModel"]

    if torch.cuda.is_available():
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        torch_dtype = torch.float32
    device_map: Optional[Dict[str, int]] = {"": 0} if torch.cuda.is_available() else None

    processor = AutoProcessor.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        cache_dir=hf_cache_dir,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        cache_dir=hf_cache_dir,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )
    model = PeftModel.from_pretrained(model, lora_dir, is_trainable=False)
    patch_adapter_weights_if_available(model, lora_dir)
    model.eval()
    return model, processor


def _move_batch_to_device(batch: Dict[str, Any], device, float_dtype):
    runtime = _lazy_import_runtime()
    torch = runtime["torch"]

    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            if torch.is_floating_point(value):
                moved[key] = value.to(device=device, dtype=float_dtype)
            else:
                moved[key] = value.to(device=device)
        else:
            moved[key] = value
    return moved


def build_messages(image_payload: Any, image_key: str, prompt_text: str, assistant_text: Optional[str] = None):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", image_key: image_payload},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    if assistant_text is not None:
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_text}],
            }
        )
    return messages


def build_processor_inputs(
    processor,
    model,
    image,
    image_path: Optional[str],
    prompt_text: str,
    assistant_text: Optional[str],
    add_generation_prompt: bool,
):
    runtime = _lazy_import_runtime()
    torch = runtime["torch"]
    if torch.cuda.is_available():
        float_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        float_dtype = torch.float32
    attempts = []
    if image is not None:
        attempts.append(("image", image))
    if image_path:
        attempts.append(("image", image_path))
        attempts.append(("url", image_path))

    last_error = None
    for image_key, image_payload in attempts:
        messages = build_messages(image_payload, image_key, prompt_text, assistant_text)
        try:
            batch = processor.apply_chat_template(
                messages,
                add_generation_prompt=add_generation_prompt,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            return _move_batch_to_device(batch, model.device, float_dtype)
        except Exception as exc:
            last_error = exc
            continue

    if last_error is not None:
        raise last_error
    raise RuntimeError("No valid image payload could be constructed for processor input.")


def prefix_token_length(processor, model, image, image_path: Optional[str], prompt_text: str) -> int:
    batch = build_processor_inputs(
        processor=processor,
        model=model,
        image=image,
        image_path=image_path,
        prompt_text=prompt_text,
        assistant_text=None,
        add_generation_prompt=True,
    )
    return int(batch["input_ids"].shape[1])


def row_col_cos(a, b):
    runtime = _lazy_import_runtime()
    F = runtime["F"]
    torch = runtime["torch"]

    if a.ndim < 2:
        sim = F.cosine_similarity(a.flatten().unsqueeze(0), b.flatten().unsqueeze(0), dim=1)
        sim = torch.nan_to_num(sim, 0.0)
        return sim, sim
    row_sim = F.cosine_similarity(a, b, dim=1)
    col_sim = F.cosine_similarity(a.T, b.T, dim=1)
    return torch.nan_to_num(row_sim, 0.0), torch.nan_to_num(col_sim, 0.0)


def infer_layer_ids(names: Sequence[str], pattern: str) -> Tuple[List[int], Optional[re.Pattern[str]]]:
    regex = re.compile(pattern)
    ids: List[int] = []
    for name in names:
        match = regex.search(name)
        if match:
            ids.append(int(match.group(1)))
    ids = sorted(set(ids))
    if ids:
        return ids, regex
    return [], None


def mark_trainable_params(model, strategy: LayerStrategy, spec: ExperimentSpec) -> List[str]:
    for _, param in model.named_parameters():
        param.requires_grad_(False)

    selected: List[str] = []
    if strategy.mode == "lora_only":
        for name, param in model.named_parameters():
            if "lora" in name.lower() and param.ndim >= 2:
                param.requires_grad_(True)
                selected.append(name)
        return selected

    if strategy.mode != "lora_last3":
        raise ValueError(f"Unknown strategy mode: {strategy.mode}")

    all_names = [name for name, _ in model.named_parameters()]
    vision_patterns = [
        r"vision_model\.encoder\.layers\.(\d+)\.",
        r"vision_tower\.vision_model\.encoder\.layers\.(\d+)\.",
        r"vision_tower\.blocks\.(\d+)\.",
        r"visual\.blocks\.(\d+)\.",
        r"visual\.model\.layers\.(\d+)\.",
    ]
    text_patterns = [
        r"language_model\.model\.layers\.(\d+)\.",
        r"model\.language_model\.model\.layers\.(\d+)\.",
        r"model\.layers\.(\d+)\.",
        r"llm\.model\.layers\.(\d+)\.",
    ]

    vision_ids: List[int] = []
    text_ids: List[int] = []
    vision_regex = None
    text_regex = None
    for pattern in vision_patterns:
        ids, regex = infer_layer_ids(all_names, pattern)
        if ids:
            vision_ids, vision_regex = ids, regex
            break
    for pattern in text_patterns:
        ids, regex = infer_layer_ids(all_names, pattern)
        if ids:
            text_ids, text_regex = ids, regex
            break

    vision_start = None
    text_start = None
    if vision_ids:
        vision_start = max(vision_ids) - (spec.vision_last_n - 1)
    if text_ids:
        text_start = max(text_ids) - (spec.text_last_m - 1)

    for name, param in model.named_parameters():
        if "lora" not in name.lower() or param.ndim < 2:
            continue
        keep = False
        lower_name = name.lower()
        if any(
            key in lower_name
            for key in ["mm_projector", "projector", "connector", "vision_proj", "multi_modal_projector"]
        ):
            keep = True
        if not keep and vision_regex is not None and vision_start is not None:
            match = vision_regex.search(name)
            if match and int(match.group(1)) >= vision_start:
                keep = True
        if not keep and text_regex is not None and text_start is not None:
            match = text_regex.search(name)
            if match and int(match.group(1)) >= text_start:
                keep = True
        if keep:
            param.requires_grad_(True)
            selected.append(name)
    return selected


def backward_and_collect(
    record: Dict[str, Any],
    media_root: str,
    model,
    processor,
    prompt_text: str,
    loss_mode: str,
    verbose_errors: bool = False,
):
    runtime = _lazy_import_runtime()
    torch = runtime["torch"]

    image_path = os.path.join(media_root, record["images"][0])
    image = load_image(image_path)
    if image is None:
        return {}

    assistant_text = extract_assistant_text(record)
    if not assistant_text:
        return {}

    try:
        full_batch = build_processor_inputs(
            processor=processor,
            model=model,
            image=image,
            image_path=image_path,
            prompt_text=prompt_text,
            assistant_text=assistant_text,
            add_generation_prompt=False,
        )
        labels = full_batch["input_ids"].clone()
        if loss_mode == "assistant_only":
            prefix_len = prefix_token_length(processor, model, image, image_path, prompt_text)
            labels[:, :prefix_len] = -100
            vocab_size = int(getattr(model.config, "vocab_size", labels.max().item() + 1))
            labels[(labels >= vocab_size) | (labels < -100)] = -100
        elif loss_mode != "full_sequence":
            raise ValueError(f"Unknown loss_mode: {loss_mode}")

        model.train()
        model.zero_grad(set_to_none=True)

        if torch.cuda.is_available():
            autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            with torch.autocast("cuda", dtype=autocast_dtype):
                outputs = model(**full_batch, labels=labels)
                loss = outputs.loss
        else:
            outputs = model(**full_batch, labels=labels)
            loss = outputs.loss

        if loss is None or not torch.isfinite(loss):
            model.zero_grad(set_to_none=True)
            model.eval()
            return {}

        loss.backward()
        grads = {}
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None and param.grad.ndim >= 2:
                # Keep gradients on the active GPU.  Moving every per-sample
                # gradient to CPU forced a device synchronization and then
                # copied it back for all subsequent similarity operations.
                grads[name] = param.grad.detach().to(torch.float32).clone()
        model.zero_grad(set_to_none=True)
        model.eval()
        return grads
    except Exception as exc:
        model.zero_grad(set_to_none=True)
        model.eval()
        if verbose_errors:
            print(f"[grad-error] {image_path}: {type(exc).__name__}: {exc}")
        return {}


def accumulate_reference_grad(
    samples: Sequence[Dict[str, Any]],
    media_root: str,
    model,
    processor,
    prompt_text: str,
    loss_mode: str,
    verbose_errors: bool = False,
):
    ref: Dict[str, Any] = {}
    valid = 0
    for record in samples:
        grad_dict = backward_and_collect(
            record=record,
            media_root=media_root,
            model=model,
            processor=processor,
            prompt_text=prompt_text,
            loss_mode=loss_mode,
            verbose_errors=verbose_errors,
        )
        if not grad_dict:
            continue
        valid += 1
        if not ref:
            ref = {key: value.clone() for key, value in grad_dict.items()}
            continue
        for key in list(ref.keys()):
            if key in grad_dict and ref[key].shape == grad_dict[key].shape:
                ref[key] += grad_dict[key]
            else:
                ref.pop(key, None)
    if valid == 0:
        raise RuntimeError("Reference gradient construction failed: 0 valid samples.")
    for key in ref:
        ref[key] /= valid
    return ref, valid


def avg_rowcol_sims(
    samples: Sequence[Dict[str, Any]],
    ref_grads: Dict[str, Any],
    media_root: str,
    model,
    processor,
    prompt_text: str,
    loss_mode: str,
    verbose_errors: bool = False,
):
    runtime = _lazy_import_runtime()
    torch = runtime["torch"]

    row_acc: Dict[str, Any] = {}
    col_acc: Dict[str, Any] = {}
    valid = 0
    for record in samples:
        grad_dict = backward_and_collect(
            record=record,
            media_root=media_root,
            model=model,
            processor=processor,
            prompt_text=prompt_text,
            loss_mode=loss_mode,
            verbose_errors=verbose_errors,
        )
        if not grad_dict:
            continue
        valid += 1
        for key, grad in grad_dict.items():
            if key not in ref_grads or ref_grads[key].shape != grad.shape:
                continue
            row_sim, col_sim = row_col_cos(grad, ref_grads[key])
            if key not in row_acc:
                row_acc[key] = row_sim.clone()
                col_acc[key] = col_sim.clone()
            else:
                if row_acc[key].shape == row_sim.shape:
                    row_acc[key] += row_sim
                if col_acc[key].shape == col_sim.shape:
                    col_acc[key] += col_sim
    if valid == 0:
        return {}, {}, 0
    for key in row_acc:
        row_acc[key] = row_acc[key] / valid
        col_acc[key] = col_acc[key] / valid
        row_acc[key] = torch.nan_to_num(row_acc[key], 0.0)
        col_acc[key] = torch.nan_to_num(col_acc[key], 0.0)
    return row_acc, col_acc, valid


def build_masks(
    spec: ExperimentSpec,
    member_rc: Tuple[Dict[str, Any], Dict[str, Any]],
    nonmember_rc: Tuple[Dict[str, Any], Dict[str, Any]],
):
    runtime = _lazy_import_runtime()
    torch = runtime["torch"]

    member_row, member_col = member_rc
    nonmember_row, nonmember_col = nonmember_rc
    row_masks: Dict[str, Any] = {}
    col_masks: Dict[str, Any] = {}
    total = 0

    for key in member_row:
        if key not in nonmember_row or key not in member_col or key not in nonmember_col:
            continue
        row_gap = torch.nan_to_num(
            (member_row[key] - nonmember_row[key]).detach().to(torch.float32),
            0.0,
        )
        col_gap = torch.nan_to_num(
            (member_col[key] - nonmember_col[key]).detach().to(torch.float32),
            0.0,
        )

        if spec.mask_mode == "tau":
            row_mask = row_gap > float(spec.sensitivity_tau)
            col_mask = col_gap > float(spec.sensitivity_tau)
        elif spec.mask_mode == "quantile_topk":
            row_mask = torch.zeros_like(row_gap, dtype=torch.bool)
            col_mask = torch.zeros_like(col_gap, dtype=torch.bool)
            if row_gap.numel() > 0:
                row_thr = torch.quantile(row_gap, float(spec.quantile_q))
                row_mask = row_gap > row_thr
                if (not row_mask.any()) and spec.topk_fallback > 0:
                    topk = min(spec.topk_fallback, row_gap.numel())
                    row_mask[torch.topk(row_gap, k=topk).indices] = True
            if col_gap.numel() > 0:
                col_thr = torch.quantile(col_gap, float(spec.quantile_q))
                col_mask = col_gap > col_thr
                if (not col_mask.any()) and spec.topk_fallback > 0:
                    topk = min(spec.topk_fallback, col_gap.numel())
                    col_mask[torch.topk(col_gap, k=topk).indices] = True
        else:
            raise ValueError(f"Unknown mask_mode: {spec.mask_mode}")

        if row_mask.any() or col_mask.any():
            row_masks[key] = row_mask
            col_masks[key] = col_mask
            total += int(row_mask.sum().item() + col_mask.sum().item())

    return row_masks, col_masks, total


def score_record(
    record: Dict[str, Any],
    media_root: str,
    ref_grads: Dict[str, Any],
    row_masks: Dict[str, Any],
    col_masks: Dict[str, Any],
    model,
    processor,
    prompt_text: str,
    loss_mode: str,
    verbose_errors: bool = False,
) -> float:
    grad_dict = backward_and_collect(
        record=record,
        media_root=media_root,
        model=model,
        processor=processor,
        prompt_text=prompt_text,
        loss_mode=loss_mode,
        verbose_errors=verbose_errors,
    )
    if not grad_dict:
        return 0.0

    sims: List[float] = []
    for key, grad in grad_dict.items():
        if (
            key not in ref_grads
            or key not in row_masks
            or key not in col_masks
            or ref_grads[key].shape != grad.shape
        ):
            continue
        row_sim, col_sim = row_col_cos(grad, ref_grads[key])
        if row_masks[key].any():
            sims.extend(row_sim[row_masks[key]].cpu().tolist())
        if col_masks[key].any():
            sims.extend(col_sim[col_masks[key]].cpu().tolist())
    return float(np.mean(sims)) if sims else 0.0


def compute_threshold_metrics(scores: np.ndarray, labels: np.ndarray, target_fpr: float):
    negative_scores = scores[labels == 0]
    if negative_scores.size == 0:
        return {
            "threshold": 0.0,
            "actual_fpr": 0.0,
            "tpr": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "accuracy": 0.0,
        }

    runtime = _lazy_import_runtime()
    accuracy_score = runtime["accuracy_score"]
    confusion_matrix = runtime["confusion_matrix"]
    precision_score = runtime["precision_score"]
    recall_score = runtime["recall_score"]

    threshold = float(np.quantile(negative_scores, 1.0 - target_fpr))
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, pred).ravel()
    return {
        "threshold": threshold,
        "actual_fpr": float(fp / (fp + tn + 1e-12)),
        "tpr": float(tp / (tp + fn + 1e-12)),
        "precision": float(precision_score(labels, pred, zero_division=0)),
        "recall": float(recall_score(labels, pred, zero_division=0)),
        "accuracy": float(accuracy_score(labels, pred)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def compute_metrics(scores: Sequence[float], labels: Sequence[int]) -> Dict[str, Any]:
    runtime = _lazy_import_runtime()
    accuracy_score = runtime["accuracy_score"]
    confusion_matrix = runtime["confusion_matrix"]
    roc_auc_score = runtime["roc_auc_score"]
    roc_curve = runtime["roc_curve"]

    scores_np = np.asarray(scores, dtype=np.float64)
    labels_np = np.asarray(labels, dtype=np.int64)
    if scores_np.size == 0:
        raise RuntimeError("Empty score list.")

    if np.allclose(scores_np, scores_np[0]):
        return {
            "auc": 0.5,
            "best_threshold": float(scores_np[0]),
            "accuracy": 0.5,
            "confusion_matrix": None,
            "roc_data": {"fpr": [], "tpr": [], "thresholds": []},
            "at_5fpr": compute_threshold_metrics(scores_np, labels_np, 0.05),
            "at_1fpr": compute_threshold_metrics(scores_np, labels_np, 0.01),
        }

    auc = float(roc_auc_score(labels_np, scores_np))
    fpr, tpr, thresholds = roc_curve(labels_np, scores_np)
    best_idx = int(np.argmax(tpr - fpr))
    best_threshold = float(thresholds[best_idx])
    pred = (scores_np >= best_threshold).astype(int)
    acc = float(accuracy_score(labels_np, pred))
    cm = confusion_matrix(labels_np, pred).tolist()

    return {
        "auc": auc,
        "best_threshold": best_threshold,
        "accuracy": acc,
        "confusion_matrix": cm,
        "roc_data": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist(),
        },
        "at_5fpr": compute_threshold_metrics(scores_np, labels_np, 0.05),
        "at_1fpr": compute_threshold_metrics(scores_np, labels_np, 0.01),
    }


def attack_split(
    spec: ExperimentSpec,
    member_records: List[Dict[str, Any]],
    nonmember_records: List[Dict[str, Any]],
):
    if spec.mask_mode == "tau":
        ref_members = member_records[: spec.ref_size]
        ref_nonmembers = nonmember_records[: spec.ref_size]
        probe_members = member_records[spec.ref_size : spec.ref_size + spec.probe_member]
        probe_nonmembers = nonmember_records[
            spec.ref_size : spec.ref_size + spec.probe_nonmember
        ]
        return {
            "ref_members": ref_members,
            "ref_nonmembers": ref_nonmembers,
            "probe_members": probe_members,
            "probe_nonmembers": probe_nonmembers,
        }

    shuffled_members = list(member_records)
    shuffled_nonmembers = list(nonmember_records)
    random.Random(spec.attack_seed).shuffle(shuffled_members)
    random.Random(spec.attack_seed).shuffle(shuffled_nonmembers)
    return {
        "ref_members": shuffled_members[: spec.calib_member],
        "ref_nonmembers": shuffled_nonmembers[: spec.calib_nonmember],
        "probe_members": shuffled_members[
            spec.calib_member : spec.calib_member + spec.probe_member
        ],
        "probe_nonmembers": shuffled_nonmembers[
            spec.calib_nonmember : spec.calib_nonmember + spec.probe_nonmember
        ],
    }


def save_strategy_outputs(
    strategy: LayerStrategy,
    spec: ExperimentSpec,
    args: RunArgs,
    metrics: Dict[str, Any],
    scores: Sequence[float],
    labels: Sequence[int],
    sample_records: Sequence[Dict[str, Any]],
    total_sensitive_dims: int,
    ref_valid: int,
    calib_member_valid: int,
    calib_nonmember_valid: int,
    model_info: Dict[str, str],
):
    result_dir = Path(args.results_dir) / spec.output_subdir
    result_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{spec.config_id}_{strategy.name}_{timestamp}"

    csv_path = result_dir / f"{stem}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["score", "label"])
        writer.writeheader()
        for score, label in zip(scores, labels):
            writer.writerow({"score": float(score), "label": int(label)})

    summary = {
        "timestamp": timestamp,
        "strategy": strategy.name,
        "spec": asdict(spec),
        "runtime": {
            "base_model": model_info["base_model"],
            "lora_dir": model_info["lora_dir"],
            "train_log": model_info.get("train_log"),
        },
        "attack": {
            "total_sensitive_dims": int(total_sensitive_dims),
            "reference_valid_samples": int(ref_valid),
            "member_calibration_valid": int(calib_member_valid),
            "nonmember_calibration_valid": int(calib_nonmember_valid),
        },
        "metrics": metrics,
    }

    summary_path = result_dir / f"{stem}_summary.json"
    samples_path = result_dir / f"{stem}_samples.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(samples_path, "w", encoding="utf-8") as f:
        json.dump(list(sample_records), f, ensure_ascii=False, indent=2)

    return {
        "csv": str(csv_path),
        "summary": str(summary_path),
        "samples": str(samples_path),
    }


def run_attack_for_strategy(
    spec: ExperimentSpec,
    args: RunArgs,
    strategy: LayerStrategy,
    model_info: Dict[str, str],
    data_paths: Dict[str, str],
) -> Dict[str, Any]:
    runtime = _lazy_import_runtime()
    torch = runtime["torch"]
    seed_everything(spec.attack_seed)

    member_records = load_json_list(data_paths["member_eval_json"])
    nonmember_records = load_json_list(data_paths["nonmember_eval_json"])
    split = attack_split(spec, member_records, nonmember_records)

    model, processor = load_model_with_lora(
        base_model_path=model_info["base_model"],
        lora_dir=model_info["lora_dir"],
        hf_cache_dir=args.hf_cache_dir,
    )
    selected = mark_trainable_params(model, strategy, spec)
    if not selected:
        raise RuntimeError(f"No trainable LoRA tensors selected for {strategy.name}.")

    ref_grads, ref_valid = accumulate_reference_grad(
        samples=split["ref_members"],
        media_root=spec.member_media_root,
        model=model,
        processor=processor,
        prompt_text=spec.attack_prompt,
        loss_mode=spec.loss_mode,
        verbose_errors=args.verbose_grad_errors,
    )

    member_row, member_col, member_valid = avg_rowcol_sims(
        samples=split["ref_members"],
        ref_grads=ref_grads,
        media_root=spec.member_media_root,
        model=model,
        processor=processor,
        prompt_text=spec.attack_prompt,
        loss_mode=spec.loss_mode,
        verbose_errors=args.verbose_grad_errors,
    )
    nonmember_row, nonmember_col, nonmember_valid = avg_rowcol_sims(
        samples=split["ref_nonmembers"],
        ref_grads=ref_grads,
        media_root=spec.nonmember_media_root,
        model=model,
        processor=processor,
        prompt_text=spec.attack_prompt,
        loss_mode=spec.loss_mode,
        verbose_errors=args.verbose_grad_errors,
    )
    if not member_row or not nonmember_row:
        raise RuntimeError(f"Calibration similarity is empty for {strategy.name}.")

    row_masks, col_masks, total_sensitive_dims = build_masks(
        spec=spec,
        member_rc=(member_row, member_col),
        nonmember_rc=(nonmember_row, nonmember_col),
    )

    scores: List[float] = []
    labels: List[int] = []
    sample_records: List[Dict[str, Any]] = []
    for record in split["probe_members"]:
        target_text = extract_assistant_text(record)
        score = score_record(
            record=record,
            media_root=spec.member_media_root,
            ref_grads=ref_grads,
            row_masks=row_masks,
            col_masks=col_masks,
            model=model,
            processor=processor,
            prompt_text=spec.attack_prompt,
            loss_mode=spec.loss_mode,
            verbose_errors=args.verbose_grad_errors,
        )
        scores.append(score)
        labels.append(1)
        sample_records.append(
            {
                "dataset": spec.dataset_name,
                "config_id": spec.config_id,
                "strategy": strategy.name,
                "image": record["images"][0],
                "image_path": os.path.join(spec.member_media_root, record["images"][0]),
                "prompt": spec.attack_prompt,
                "target_text": target_text,
                "score": float(score),
                "label": 1,
                "split": "probe_member",
            }
        )
    for record in split["probe_nonmembers"]:
        target_text = extract_assistant_text(record)
        score = score_record(
            record=record,
            media_root=spec.nonmember_media_root,
            ref_grads=ref_grads,
            row_masks=row_masks,
            col_masks=col_masks,
            model=model,
            processor=processor,
            prompt_text=spec.attack_prompt,
            loss_mode=spec.loss_mode,
            verbose_errors=args.verbose_grad_errors,
        )
        scores.append(score)
        labels.append(0)
        sample_records.append(
            {
                "dataset": spec.dataset_name,
                "config_id": spec.config_id,
                "strategy": strategy.name,
                "image": record["images"][0],
                "image_path": os.path.join(spec.nonmember_media_root, record["images"][0]),
                "prompt": spec.attack_prompt,
                "target_text": target_text,
                "score": float(score),
                "label": 0,
                "split": "probe_nonmember",
            }
        )

    metrics = compute_metrics(scores, labels)
    saved = save_strategy_outputs(
        strategy=strategy,
        spec=spec,
        args=args,
        metrics=metrics,
        scores=scores,
        labels=labels,
        sample_records=sample_records,
        total_sensitive_dims=total_sensitive_dims,
        ref_valid=ref_valid,
        calib_member_valid=member_valid,
        calib_nonmember_valid=nonmember_valid,
        model_info=model_info,
    )

    del model, processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return {
        "strategy": strategy.name,
        "metrics": metrics,
        "sensitive_dims": total_sensitive_dims,
        "outputs": saved,
    }


def run_full_pipeline(spec: ExperimentSpec, args: RunArgs) -> List[Dict[str, Any]]:
    print(f"\n{'=' * 80}")
    print(f"Experiment: {spec.dataset_name} | {spec.config_id}")
    print(f"{'=' * 80}")

    ensure_dir(args.work_dir)
    ensure_dir(args.results_dir)
    ensure_dir(args.adapter_store)
    seed_everything(spec.train_seed)

    data_paths = build_training_and_attack_sets(spec, args.work_dir)
    model_info = train_with_llamafactory(spec, args, data_paths)

    results: List[Dict[str, Any]] = []
    for strategy in spec.layer_strategies:
        print(f"\n--- Attack strategy: {strategy.name} ({strategy.mode}) ---")
        result = run_attack_for_strategy(
            spec=spec,
            args=args,
            strategy=strategy,
            model_info=model_info,
            data_paths=data_paths,
        )
        metrics = result["metrics"]
        print(
            f"AUC={metrics['auc']:.4f}  "
            f"TPR@5%FPR={metrics['at_5fpr']['tpr']:.4f}  "
            f"TPR@1%FPR={metrics['at_1fpr']['tpr']:.4f}  "
            f"sensitive_dims={result['sensitive_dims']}"
        )
        results.append(result)
    return results


import argparse
from pathlib import Path



DEFAULT_BASE_DIR = str(Path(__file__).resolve().parents[1])
DEFAULT_DATA_SCRATCH = DEFAULT_BASE_DIR
DEFAULT_LLAMA_REPO_CANDS = [
    f"{DEFAULT_BASE_DIR}/models/LLaMA-Factory",
    f"{DEFAULT_BASE_DIR}/LLaMA-Factory",
]


def resolve_llama_repo(candidate: str | None) -> str:
    if candidate:
        return candidate
    for path in DEFAULT_LLAMA_REPO_CANDS:
        if Path(path).is_dir():
            return path
    return DEFAULT_LLAMA_REPO_CANDS[0]


def parse_args() -> RunArgs:
    parser = argparse.ArgumentParser(
        description="InternVL3-1B-hf LoRA + GradAudit pipeline for MedTrinity."
    )
    parser.add_argument("--base_dir", default=DEFAULT_BASE_DIR)
    parser.add_argument("--llama_repo", default=None)
    parser.add_argument("--base_model", default="OpenGVLab/InternVL3-1B-hf")
    parser.add_argument("--hf_cache_dir", default=f"{DEFAULT_BASE_DIR}/models/hf_cache")
    parser.add_argument(
        "--work_dir",
        default=f"{DEFAULT_DATA_SCRATCH}/experiment_workspace/internvl3_med",
    )
    parser.add_argument(
        "--adapter_store",
        default=f"{DEFAULT_DATA_SCRATCH}/adapters",
    )
    parser.add_argument(
        "--results_dir",
        default=f"{DEFAULT_BASE_DIR}/results/internvl3",
    )
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--verbose_grad_errors", action="store_true")
    ns = parser.parse_args()
    return RunArgs(
        base_dir=ns.base_dir,
        llama_repo=resolve_llama_repo(ns.llama_repo),
        base_model=ns.base_model,
        hf_cache_dir=ns.hf_cache_dir,
        work_dir=ns.work_dir,
        adapter_store=ns.adapter_store,
        results_dir=ns.results_dir,
        fresh=ns.fresh,
        skip_train=ns.skip_train,
        verbose_grad_errors=ns.verbose_grad_errors,
    )


def build_spec() -> ExperimentSpec:
    return ExperimentSpec(
        config_id="internvl3_medtrinity_1k_1b",
        dataset_name="MedTrinity",
        train_mode="member_eval_flat",
        member_json=f"{DEFAULT_BASE_DIR}/data/frozen_dataset/medtrinity_mia/member_eval_1000.json",
        nonmember_json=f"{DEFAULT_BASE_DIR}/data/frozen_dataset/medtrinity_mia/nonmember_eval_1000.json",
        member_media_root=f"{DEFAULT_BASE_DIR}/data/frozen_dataset/medtrinity_mia/member_5k",
        nonmember_media_root=f"{DEFAULT_BASE_DIR}/data/frozen_dataset/medtrinity_mia/nonmember_5k",
        train_prompt="As a medical vision-language model, please analyze the image and generate a concise, clinically sound diagnostic summary.",
        # The attack must reproduce the exact multimodal sequence used for SFT.
        attack_prompt="As a medical vision-language model, please analyze the image and generate a concise, clinically sound diagnostic summary.",
        output_subdir="medtrinity",
        train_size=1000,
        train_seed=42,
        train_epochs=5.0,
        learning_rate=1e-4,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        lora_rank=64,
        lora_alpha=128,
        max_samples=1000,
        ref_size=200,
        probe_member=800,
        probe_nonmember=800,
        attack_seed=42,
        loss_mode="assistant_only",
        mask_mode="tau",
        # Medical-domain adaptation: supervise the diagnostic caption only.
        # Strict full-sequence tau=0.10 and full-sequence tau=0.05 results are
        # retained in their timestamped summaries for transparent comparison.
        sensitivity_tau=0.10,
        layer_strategies=[LayerStrategy(name="only_lora", mode="lora_only")],
    )


if __name__ == "__main__":
    run_full_pipeline(build_spec(), parse_args())
