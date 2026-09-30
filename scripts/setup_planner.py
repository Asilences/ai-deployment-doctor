"""Download one pinned planner asset, without modifying the target model or runtime."""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from inference_lab.core import dump
from planning_service import validate_profile
from setup_windows import fetch, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('profile', type=Path)
    args = parser.parse_args()
    profile = validate_profile(json.loads(args.profile.read_text(encoding='utf-8-sig')), ROOT)
    model = (ROOT / profile['model_path']).resolve()
    started = time.monotonic()
    url = 'https://huggingface.co/' + profile['model'] + '/resolve/' + profile['model_revision'] + '/' + model.name
    fetch(url, model, profile['model_sha256'])
    if model.stat().st_size != profile['model_size_bytes']:
        raise ValueError('Published model size mismatch')
    digest = sha256(model)
    dump(model.with_suffix(model.suffix + '.manifest.json'), {
        'model': profile['model'], 'model_revision': profile['model_revision'], 'model_sha256': digest})
    dump(ROOT / 'setup_records' / (profile['service_id'] + '.json'), {
        'profile': profile, 'download_url': url, 'verified_sha256': digest,
        'verified_bytes': model.stat().st_size, 'asset_setup_seconds': time.monotonic() - started,
        'note': 'Asset preparation cost; excluded from model generation and session costs'})
    print('Verified planner asset:', profile['service_id'], flush=True)


if __name__ == '__main__':
    main()
