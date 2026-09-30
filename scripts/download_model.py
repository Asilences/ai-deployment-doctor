"""Download the experiment model at its frozen revision; run in the Linux venv."""
import json
from pathlib import Path

from huggingface_hub import snapshot_download

root = Path(__file__).resolve().parents[1]
config = json.loads((root / 'configs/local.json').read_text())
destination = root / config['model_path']
snapshot_download(repo_id=config['model'], revision=config['model_revision'], local_dir=destination,
                  allow_patterns=['*.json', '*.safetensors', 'merges.txt', 'vocab.json', '*.model'])
(destination / 'download_manifest.json').write_text(json.dumps({
    'model': config['model'], 'revision': config['model_revision']}, indent=2), encoding='utf-8')
print('Pinned model downloaded to ' + str(destination))
