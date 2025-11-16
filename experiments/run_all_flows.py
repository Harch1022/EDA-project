#!/usr/bin/env python3
import subprocess
import yaml
import os
import sys
from typing import List

def run_design(design: str, project_root: str) -> int:
    env = os.environ.copy()
    env["DESIGN_NAME"] = design
    env["TOP_MODULE"] = env.get("TOP_MODULE", design)
    cmd = ["bash", f"{project_root}/eda_flow/run_flow.sh"]
    print(f"[run_all] Running: DESIGN={design}")
    return subprocess.call(cmd, env=env)

def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    cfg_path = os.path.join(project_root, "configs", "benches.yaml")
    if not os.path.exists(cfg_path):
        print(f"ERROR: benches.yaml not found: {cfg_path}")
        sys.exit(1)
    designs: List[str] = yaml.safe_load(open(cfg_path, "r"))
    if not isinstance(designs, list) or not designs:
        print("ERROR: benches.yaml must be a list of design names.")
        sys.exit(1)
    for d in designs:
        code = run_design(d, project_root)
        if code != 0:
            print(f"[run_all] FAILED on {d} with code {code}. Abort.")
            sys.exit(code)
    print("[run_all] All designs completed.")

if __name__ == "__main__":
    main()