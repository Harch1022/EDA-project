#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ORFS="$PROJECT_ROOT/tools/OpenROAD-flow-scripts/flow/platforms/nangate45"
LIB_SRC="$ORFS/lib/NangateOpenCellLibrary_typical.lib"
LEF_TECH_SRC="$ORFS/lef/NangateOpenCellLibrary.tech.lef"
LEF_STD_SRC="$ORFS/lef/NangateOpenCellLibrary.macro.lef"

mkdir -p "$PROJECT_ROOT/eda_flow/lib"
ln -sf "$LIB_SRC"      "$PROJECT_ROOT/eda_flow/lib/typical.lib"
ln -sf "$LEF_TECH_SRC" "$PROJECT_ROOT/eda_flow/lib/tech.lef"
ln -sf "$LEF_STD_SRC"  "$PROJECT_ROOT/eda_flow/lib/stdcells.lef"

echo "Symlinks created under eda_flow/lib -> ORFS Nangate45."
ls -l "$PROJECT_ROOT/eda_flow/lib"