#!/usr/bin/env python3
"""Create HF collections for ATLAS model families - validated, batch-safe."""
from huggingface_hub import HfApi
import sys, time

api = HfApi()

COLLECTIONS = [
    {
        "title": "Falcon3 - ATLAS TQ1_0",
        "description": "TII Falcon3 ternary-quantized ATLAS TQ1_0 models. Instruct (1B/3B/7B/10B) + Base (3B/7B/10B), CPU inference.",
        "items": [
            ("xxxn3m3s1sxxx/Falcon3-1B-Instruct-1.58bit-ATLAS", "1B Instruct"),
            ("xxxn3m3s1sxxx/Falcon3-3B-Instruct-1.58bit-ATLAS", "3B Instruct"),
            ("xxxn3m3s1sxxx/Falcon3-7B-Instruct-1.58bit-ATLAS", "7B Instruct"),
            ("xxxn3m3s1sxxx/Falcon3-10B-Instruct-1.58bit-ATLAS", "10B Instruct"),
            ("xxxn3m3s1sxxx/Falcon3-3B-Base-1.58bit-ATLAS", "3B Base"),
            ("xxxn3m3s1sxxx/Falcon3-7B-Base-1.58bit-ATLAS", "7B Base"),
            ("xxxn3m3s1sxxx/Falcon3-10B-Base-1.58bit-ATLAS", "10B Base"),
        ],
    },
    {
        "title": "Bonsai - ATLAS TQ1_0",
        "description": "Prism ML Bonsai (Qwen3) ternary-quantized ATLAS TQ1_0. 1.7B/4B/8B, CPU inference.",
        "items": [
            ("xxxn3m3s1sxxx/Ternary-Bonsai-1.7B-ATLAS", "1.7B"),
            ("xxxn3m3s1sxxx/Ternary-Bonsai-4B-ATLAS", "4B"),
            ("xxxn3m3s1sxxx/Ternary-Bonsai-8B-ATLAS", "8B"),
        ],
    },
    {
        "title": "BitNet - ATLAS TQ1_0",
        "description": "BitNet b1.58 2B-4T (Microsoft) ATLAS TQ1_0 model. Ternary CPU inference.",
        "items": [
            ("xxxn3m3s1sxxx/BitNet-2B4T-b1.58-ATLAS", "2B-4T"),
        ],
    },
    {
        "title": "BitCPM-CANN - ATLAS TQ1_0",
        "description": "BitCPM CANN ternary-quantized ATLAS TQ1_0 models. 0.5B/1B/3B/8B, CPU inference.",
        "items": [
            ("xxxn3m3s1sxxx/BitCPM-CANN-0.5B-ATLAS", "0.5B"),
            ("xxxn3m3s1sxxx/BitCPM-CANN-1B-ATLAS", "1B"),
            ("xxxn3m3s1sxxx/BitCPM-CANN-3B-ATLAS", "3B"),
            ("xxxn3m3s1sxxx/BitCPM-CANN-8B-ATLAS", "8B"),
        ],
    },
    {
        "title": "TriLM + Llama3 - ATLAS TQ1_0",
        "description": "TriLM (99M-3.9B) + Llama3-8B-1.58 — all ternary-quantized ATLAS TQ1_0, CPU inference.",
        "items": [
            ("xxxn3m3s1sxxx/TriLM-99M-ATLAS", "99M"),
            ("xxxn3m3s1sxxx/TriLM-190M-ATLAS", "190M"),
            ("xxxn3m3s1sxxx/TriLM-390M-ATLAS", "390M"),
            ("xxxn3m3s1sxxx/TriLM-560M-ATLAS", "560M"),
            ("xxxn3m3s1sxxx/TriLM-830M-ATLAS", "830M"),
            ("xxxn3m3s1sxxx/TriLM-1.1B-ATLAS", "1.1B"),
            ("xxxn3m3s1sxxx/TriLM-1.5B-ATLAS", "1.5B"),
            ("xxxn3m3s1sxxx/TriLM-2.4B-ATLAS", "2.4B"),
            ("xxxn3m3s1sxxx/TriLM-3.9B-ATLAS", "3.9B"),
            ("xxxn3m3s1sxxx/Llama3-8B-1.58-ATLAS", "Llama3-8B"),
        ],
    },
]


def validate():
    """Dry-run: check titles, descriptions, model existence."""
    errors = 0
    for col in COLLECTIONS:
        title = col["title"]
        desc = col["description"]
        if len(title) > 255:
            print(f"FAIL: Title too long ({len(title)}): {title}")
            errors += 1
        if len(desc) > 150:
            print(f"FAIL: Description too long ({len(desc)}): {desc}")
            errors += 1
        for repo_id, _note in col["items"]:
            try:
                api.list_repo_files(repo_id, repo_type="model")
                print(f"  OK  {repo_id}")
            except Exception as e:
                print(f"FAIL  {repo_id}: {e}")
                errors += 1
    if errors:
        print(f"\n{errors} error(s) found.")
        sys.exit(1)
    print("\nAll validation passed.\n")


def create_all(dry_run=False):
    """Create all collections (dry_run=True = skip actual POSTs)."""
    if dry_run:
        validate()
        for col in COLLECTIONS:
            print(f'Would create: {col["title"]} ({len(col["items"])} items)')
        return

    validate()
    for col in COLLECTIONS:
        title = col["title"]
        try:
            result = api.create_collection(
                title=title,
                description=col["description"],
                namespace="xxxn3m3s1sxxx",
                private=False,
                exists_ok=False,
            )
            slug = result.slug
            last = slug[slug.rfind("/") + 1:]
            print(f"  + Created: {title}")
            print(f"    https://huggingface.co/collections/xxxn3m3s1sxxx/{last}")

            for repo_id, note in col["items"]:
                time.sleep(1)
                api.add_collection_item(slug, repo_id, item_type="model", note=note)
                print(f"    + {repo_id}")

        except Exception as e:
            err_str = str(e)
            if "409" in err_str or "already exists" in err_str.lower():
                print(f"  ~ Exists: {title}")
            else:
                print(f"  FAIL: {title} - {e}")
                return


if __name__ == "__main__":
    dry = "--dry-run" in sys.argv
    create_all(dry_run=dry)
