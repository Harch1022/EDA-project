#!/usr/bin/env bash
set -euo pipefail
echo "=== Versions ==="
echo "[OS]" && uname -a || true
echo "[Python]" && python3 -V || true
echo "[Pip]" && pip -V || true
echo "[Yosys]" && (yosys -V || true)
echo "[OpenROAD]" && (openroad -version || openroad -help || true)
echo "[Python packages]"
python3 - <<'PY'
import pkgutil
import importlib
mods = ["torch","dgl","numpy","scipy","pandas","yaml","networkx","pytest","tqdm"]
for m in mods:
    try:
        mod = importlib.import_module(m if m!="yaml" else "yaml")
        v = getattr(mod, "__version__", "unknown")
        print(f"{m}: {v}")
    except Exception as e:
        print(f"{m}: not installed")
PY