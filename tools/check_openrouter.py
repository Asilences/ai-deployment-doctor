"""Explicit OpenRouter preflight. Reads only this project's .env."""
import argparse
import json
import os
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, build_opener, HTTPRedirectHandler

class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None

def settings():
    config = {}
    env_file = Path(__file__).resolve().parents[1] / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                config[key.strip()] = value.strip().strip("\"'")
    for name in ("OPENROUTER_API_KEY", "OPENROUTER_BASE_URL", "OPENROUTER_MODEL"):
        if name in os.environ:
            config[name] = os.environ[name]
    return config

def request(config, path, payload=None):
    key = config.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise ValueError("OPENROUTER_API_KEY is missing; configure it locally.")
    base = config.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
    parsed = urlsplit(base)
    if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.query or parsed.fragment:
        raise ValueError("Use an HTTPS API base without embedded credentials, query, or fragment.")
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = Request(base + path, data=data, headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    try:
        with build_opener(NoRedirect).open(req, timeout=60) as response:
            return json.load(response)
    except HTTPError as error:
        # Do not print remote bodies or request headers, which may contain secrets.
        raise RuntimeError("API returned HTTP " + str(error.code)) from None
    except URLError:
        raise RuntimeError("Connection failed; check DNS, HTTPS access, and configured endpoint.") from None

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completion", action="store_true", help="Send one billable model smoke test")
    args = parser.parse_args()
    config = settings()
    try:
        if args.completion:
            model = config.get("OPENROUTER_MODEL", "").strip()
            if not model:
                raise ValueError("Set an approved OPENROUTER_MODEL before running a completion.")
            result = request(config, "/chat/completions", {"model": model, "messages": [{"role": "user", "content": "Reply with OK only."}], "max_tokens": 64})
            choices = result.get("choices", [])
            valid = bool(choices and choices[0].get("message", {}).get("content"))
            if not valid:
                raise ValueError("API response has no nonempty assistant content; inspect model compatibility locally.")
            usage = result.get("usage", {})
            print(json.dumps({"completion_received": True, "model": result.get("model", model), "usage": {k: usage.get(k) for k in ("prompt_tokens", "completion_tokens", "total_tokens", "cost")}}, indent=2))
        else:
            result = request(config, "/key")
            data = result.get("data")
            if not isinstance(data, dict):
                raise ValueError("Unexpected key-status response. A laboratory proxy may not implement /key.")
            print(json.dumps({"key_status_received": True, "is_management_key": data.get("is_management_key"), "limit_remaining": data.get("limit_remaining"), "is_free_tier": data.get("is_free_tier")}, indent=2))
            if data.get("is_management_key"):
                raise ValueError("A management key cannot be used for inference; request an inference key.")
    except (ValueError, RuntimeError):
        # Only our explicit errors may be printed; no credential or remote body is echoed.
        error = sys.exc_info()[1]
        if isinstance(error, json.JSONDecodeError):
            print("ERROR: API response was not JSON.", file=sys.stderr)
        else:
            print("ERROR: " + str(error), file=sys.stderr)
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
