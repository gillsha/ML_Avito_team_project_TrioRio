"""Download and verify the upstream Qwen snapshot used by the experiment."""
import argparse
from pathlib import Path

from huggingface_hub import snapshot_download
from rank_a08 import MODEL, REVISION, WEIGHT_SHA256, file_hash

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('directory', type=Path)
args = parser.parse_args()
snapshot_download(MODEL, revision=REVISION, local_dir=args.directory,
                  allow_patterns=['*.safetensors', '*.json', 'merges.txt', 'LICENSE'])
for shard, expected in WEIGHT_SHA256.items():
    if file_hash(args.directory / shard) != expected:
        raise SystemExit(f'SHA-256 mismatch for {shard}')
