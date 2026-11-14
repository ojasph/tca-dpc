#!/bin/bash
# Prepares ./dpcc (submodule: upstream DPCC @c2f25ae) with the TCA-DPC patch and tilt sampler, and links ./d3il (submodule: D3IL @1d9c718)
# into it. TCA-DPC's own folders are symlinked into the DPCC tree, so edits in this repo take effect directly.
set -e
cd "$(dirname "$0")"
git submodule update --init dpcc d3il          # not --recursive: DPCC's own d3il pointer has no URL, ./d3il replaces it
if git -C dpcc diff --quiet; then
  git -C dpcc apply ../patches/dpcc.patch
else
  echo "dpcc already modified, skipping patch (reset with: git -C dpcc checkout . )"
fi
ln -sfn ../../../tca_dpc/tilt.py dpcc/diffuser/sampling/tilt.py
ln -sfn ../../configs dpcc/config/tca_dpc
ln -sfn ../../scripts dpcc/scripts/tca_dpc
ln -sfn ../analysis   dpcc/analysis
[ -L dpcc/d3il ] || { rmdir dpcc/d3il 2>/dev/null || true; ln -s ../d3il dpcc/d3il; }
echo "Done. Next: cd dpcc && pip install -r requirements.txt (see README for the D3IL dataset)."
