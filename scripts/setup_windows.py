"""Fetch pinned official llama.cpp Vulkan build and pinned GGUF model."""
import hashlib
import argparse
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'vendor/llama.cpp-b11138.zip'
DEST = ROOT / 'vendor/llama.cpp-b11138'
BUILD_URL = 'https://github.com/ggml-org/llama.cpp/releases/download/b11138/llama-b11138-bin-win-vulkan-x64.zip'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(url, destination, expected=None):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and (expected is None or sha256(destination) == expected):
        print('Already present: ' + str(destination), flush=True)
        return
    partial = destination.with_suffix(destination.suffix + '.partial')
    if expected and partial.exists() and sha256(partial) == expected:
        partial.replace(destination)
        print('Verified previous download: ' + str(destination), flush=True)
        return
    req = urllib.request.Request(url, headers={'User-Agent': 'InferenceLab/0.1'})
    print('Downloading: ' + destination.name, flush=True)
    with urllib.request.urlopen(req, timeout=120) as response, partial.open('wb') as target:
        length = int(response.headers.get('Content-Length', '0'))
        total = 0
        while chunk := response.read(1024 * 1024):
            total += len(chunk)
            target.write(chunk)
            if total % (25 * 1024 * 1024) < len(chunk):
                print(f'  {total / 2**20:.0f} / {length / 2**20:.0f} MiB', flush=True)
    if expected and sha256(partial) != expected:
        raise RuntimeError('Downloaded file failed its published SHA-256 check')
    partial.replace(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/windows-1.5b.json')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8-sig'))
    model = (ROOT / config['model_path']).resolve()
    model_url = ('https://huggingface.co/' + config['model'] + '/resolve/'
                 + config['model_revision'] + '/' + model.name)
    fetch(BUILD_URL, ARCHIVE)
    DEST.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ARCHIVE) as z:
        for entry in z.infolist():
            # Extract only the flat official binary package, never arbitrary paths.
            if entry.is_dir():
                continue
            name = Path(entry.filename).name
            allowed = name.lower().endswith(('.exe', '.dll')) or name == 'LICENSE-LLVM-OpenMP'
            if not name or name != entry.filename or not allowed:
                raise RuntimeError('Unexpected release archive layout')
            output = DEST / name
            if not output.exists():
                with z.open(entry) as source, output.open('wb') as target:
                    shutil.copyfileobj(source, target)
    if not (DEST / 'llama-server.exe').is_file():
        raise RuntimeError('Release archive is missing llama-server.exe')
    fetch(model_url, model, config['model_sha256'])
    build_manifest = {'build_tag': config['llama_cpp_version'], 'archive_sha256': sha256(ARCHIVE)}
    (DEST / 'download_manifest.json').write_text(json.dumps(build_manifest, indent=2), encoding='utf-8')
    model_manifest = {'model': config['model'], 'model_revision': config['model_revision'],
                      'model_sha256': sha256(model)}
    model.with_suffix(model.suffix + '.manifest.json').write_text(
        json.dumps(model_manifest, indent=2), encoding='utf-8')
    print('Windows inference assets ready.', flush=True)


if __name__ == '__main__':
    main()
