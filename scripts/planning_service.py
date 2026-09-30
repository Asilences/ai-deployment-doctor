"""Independent local planner identity. Target settings never configure this service."""
import re
from pathlib import Path

from inference_lab.core import fingerprint
from inference_lab.proposal_replay import require

FIELDS = {'version', 'service_id', 'backend', 'model', 'model_revision', 'model_sha256',
          'model_size_bytes', 'model_path', 'llama_cpp_version', 'llama_cpp_binary',
          'port', 'max_model_len', 'batch_size', 'gpu_layers', 'cpu_threads', 'startup_timeout_s'}


def validate_profile(profile, root):
    require(set(profile) == FIELDS, 'Planner profile fields differ from the explicit service schema')
    require(profile['version'] == 'local-planning-service-v1' and profile['backend'] == 'llama_cpp',
            'Unsupported planning service')
    require(re.fullmatch(r'[a-z0-9-]{1,48}', profile['service_id']) is not None, 'Invalid service ID')
    require(re.fullmatch(r'[0-9a-f]{40}', profile['model_revision']) is not None, 'Unpinned revision')
    require(re.fullmatch(r'[0-9a-f]{64}', profile['model_sha256']) is not None, 'Invalid model checksum')
    require(re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', profile['model']) is not None, 'Invalid model repository')
    for field, low, high in [('port', 1024, 65535), ('max_model_len', 300, 32768),
                             ('batch_size', 1, 4096), ('gpu_layers', 0, 999),
                             ('cpu_threads', 1, 128), ('startup_timeout_s', 1, 180),
                             ('model_size_bytes', 1, 100_000_000_000)]:
        require(type(profile[field]) is int and low <= profile[field] <= high, 'Invalid ' + field)
    require(Path(profile['model_path']).suffix == '.gguf', 'Expected a GGUF planner model')
    for key, directory in [('model_path', 'models'), ('llama_cpp_binary', 'vendor')]:
        path = Path(profile[key])
        require(not path.is_absolute() and (root / path).resolve().is_relative_to((root / directory).resolve()),
                'Planner path escapes ' + directory)
    return profile


def service_identity(profile, root):
    validate_profile(profile, root)
    return {'profile': profile.copy(), 'profile_sha256': fingerprint(profile),
            'endpoint': 'http://127.0.0.1:' + str(profile['port']) + '/v1/chat/completions',
            'model_selector': str((root / profile['model_path']).resolve())}


def planner_payload(brief, target_settings, profile, root):
    # Build the target evidence and schema unchanged, then explicitly select the planning model.
    from inference_lab.proposal_replay import payload_for
    payload = payload_for(brief, target_settings, root)
    payload['model'] = service_identity(profile, root)['model_selector']
    return payload
