#!/usr/bin/env python3
"""Download TriLM + Llama3 atlas files from HF Hub and test generation."""
import os, sys, json, tempfile, logging, argparse
from huggingface_hub import HfApi
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from atlas_infer import AtlasModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger("test_trilm")

api = HfApi()

MODELS = [
    "TriLM-99M-ATLAS",
    "TriLM-190M-ATLAS",
    "TriLM-390M-ATLAS",
    "TriLM-560M-ATLAS",
    "TriLM-830M-ATLAS",
    "TriLM-1.1B-ATLAS",
    "TriLM-1.5B-ATLAS",
    "TriLM-2.4B-ATLAS",
    "TriLM-3.9B-ATLAS",
    "Llama3-8B-1.58-ATLAS",
]

def test_one(name, repo, atlas_file, results):
    tmp = tempfile.mkdtemp(prefix="atlas_test_")
    local_path = os.path.join(tmp, atlas_file)
    try:
        log.info(f"  Downloading {atlas_file}...")
        api.hf_hub_download(repo, atlas_file, local_dir=tmp, local_dir_use_symlinks=False)
        size_gb = os.path.getsize(local_path) / (1024**3)

        log.info(f"  Loading model ({size_gb:.2f} GB)...")
        model = AtlasModel(local_path)
        model.set_seed(42)
        model.set_num_threads(4)

        log.info(f"  Generating...")
        out = model.generate_c("The capital of France is", max_new_tokens=30, temperature=0.7, top_k=40)
        out = out.strip()
        log.info(f"  Output: \"{out}\"")

        # Check for coherence
        passed = bool(out) and len(out) > 3 and "Paris" in out
        results.append((name, size_gb, out, "PASS" if passed else "WARN"))
        if not passed:
            log.warning(f"  Coherence warning: no 'Paris' in output")
    except Exception as e:
        results.append((name, 0, str(e), "FAIL"))
        log.error(f"  FAILED: {e}")
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
        log.info(f"  Cleaned up")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-at", default=None)
    parser.add_argument("--quick", action="store_true", help="Skip Llama3-8B")
    args = parser.parse_args()

    results = []
    started = not args.start_at
    for name in MODELS:
        if args.start_at and args.start_at not in name:
            if not started:
                continue
        if not started and args.start_at and args.start_at in name:
            started = True

        repo = f"xxxn3m3s1sxxx/{name}"
        atlas_file = f"{name}.tq1.atlas"
        log.info(f"\n{'='*60}")
        log.info(f"Testing: {name}")
        log.info(f"{'='*60}")

        files = list(api.list_repo_files(repo))
        if atlas_file not in files:
            log.error(f"  {atlas_file} not found in repo! Skipping.")
            results.append((name, 0, "file not found", "SKIP"))
            continue

        test_one(name, repo, atlas_file, results)

    log.info(f"\n{'='*60}")
    log.info(f"RESULTS")
    log.info(f"{'='*60}")
    for name, size, out, status in results:
        icon = { "PASS": "PASS", "WARN": "WARN", "FAIL": "FAIL", "SKIP": "SKIP" }.get(status, "????")
        print(f"  [{icon}] {name} ({size:.2f} GB): \"{out[:60]}\"")

    n_fail = sum(1 for r in results if r[3] == "FAIL")
    n_skip = sum(1 for r in results if r[3] == "SKIP")
    log.info(f"  {len(results)} models, {len(results)-n_fail-n_skip} pass, {n_fail} fail, {n_skip} skip")

if __name__ == "__main__":
    main()
