"""Read-only server report. No packages, services, or clusters are changed."""
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

report = {"os": platform.system(), "architecture": platform.machine(), "python": platform.python_version(), "logical_cpus": os.cpu_count(), "disk_free_gib": round(shutil.disk_usage(Path.cwd()).free / 2**30, 1)}
if Path("/proc/meminfo").exists():
    line = next(x for x in Path("/proc/meminfo").read_text().splitlines() if x.startswith("MemTotal:"))
    report["host_ram_gib"] = round(int(line.split()[1]) / 2**20, 1)
for rel in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
    p = Path(rel)
    if p.exists():
        val = p.read_text().strip()
        report["container_memory_limit"] = val
        break
report["tools"] = {name: bool(shutil.which(name)) for name in ("docker", "git", "kubectl", "helm", "kind", "uv")}
report["docker_daemon_reachable"] = False
if report["tools"]["docker"]:
    try:
        p = subprocess.run(["docker", "info", "--format", "{{.ServerVersion}}"], capture_output=True, text=True, timeout=15)
        report["docker_daemon_reachable"] = p.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        pass
report["sregym_ready"] = False
report["note"] = "Inventory only. SREGym still requires version checks, image/network access, sufficient resources and a real test run."
print(json.dumps(report, indent=2))
