"""Download the model assets pinned by configs/models.json."""
from __future__ import annotations
import argparse, json, os, subprocess
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs/models.json").read_text())

def snapshot(item: dict, target: Path, token: str | None, cache: str) -> None:
    snapshot_download(repo_id=item["repository"], revision=item.get("revision"), token=token,
                      cache_dir=cache, local_dir=target)

def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--model", default="all", choices=["all", *CONFIG]); p.add_argument("--cache-dir", default=str(ROOT / "models/hf_cache")); a = p.parse_args()
    chosen = CONFIG if a.model == "all" else {a.model: CONFIG[a.model]}; token = os.environ.get("HF_TOKEN")
    for name, item in chosen.items():
        print(f"Downloading {name}")
        if name == "minigpt_v2":
            ckpt = ROOT / "models/minigpt_v2/checkpoints"; ckpt.mkdir(parents=True, exist_ok=True)
            hf_hub_download(repo_id=item["repository"], repo_type="space", filename=item["checkpoint_file"], token=token, local_dir=ckpt, cache_dir=a.cache_dir)
            snapshot_download(repo_id=item["llm_repository"], revision=item["llm_revision"], token=token, local_dir=ROOT / "models/minigpt_v2/llm/Llama-2-7b-chat-hf", cache_dir=a.cache_dir)
            code = ROOT / "models/minigpt_v2/MiniGPT-4"
            if not code.exists(): subprocess.run(["git", "clone", item["code_repository"], str(code)], check=True)
            subprocess.run(["git", "-C", str(code), "checkout", item["code_commit"]], check=True)
        elif name == "smolvlm2": snapshot(item, ROOT / "models/SmolVLM2-2.2B-Instruct", token, a.cache_dir)
        elif name == "minicpm_v2": snapshot(item, ROOT / "models/MiniCPM-V-2", token, a.cache_dir)
        elif name == "llava_med": snapshot(item, ROOT / "models/llava-med-v1.5-mistral-7b-hf", token, a.cache_dir)
        elif name == "qwen2vl_med": snapshot(item, ROOT / "models/Qwen2-VL-2B-Instruct", token, a.cache_dir)
        elif name == "internvl3_med": snapshot(item, ROOT / "models/InternVL3-1B-hf", token, a.cache_dir)
    if any(name in chosen for name in ("qwen2vl_med", "internvl3_med")):
        factory = ROOT / "models/LLaMA-Factory"
        if not factory.exists(): subprocess.run(["git", "clone", "https://github.com/hiyouga/LLaMA-Factory.git", str(factory)], check=True)
        subprocess.run(["git", "-C", str(factory), "checkout", "57354fc9904fe36daa6ddd9010d4936d84188ddb"], check=True)
    print("Requested model assets are ready under models/.")

if __name__ == "__main__": main()
