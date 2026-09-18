#!/bin/sh
# Finish moving the per-PR demo notebooks from doc/notebooks/ to notebooks/ and add
# external-documentation links, one commit per branch. fmatch-1/2/3 are already done;
# fmatch-4 is prepped and staged (this script just commits it). Safe to re-run: branches
# already committed are skipped.
#
# Run from the repo root:  sh doc/notebook_move/run_remaining.sh
set -e

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"
PY="$REPO_ROOT/.venv/bin/python"
JUPYTER="$REPO_ROOT/.venv/bin/jupyter"
PATCH="$REPO_ROOT/doc/notebook_move/add_links.py"

do_branch() {
    branch="$1"; nb="$2"; msg="$3"
    echo ""
    echo "=== $branch ==="
    git checkout -q "$branch"
    if git ls-files --error-unmatch "notebooks/$nb" >/dev/null 2>&1; then
        echo "already committed — skipping"
        return
    fi
    if [ -f "doc/notebooks/$nb" ]; then
        mkdir -p notebooks
        git mv "doc/notebooks/$nb" "notebooks/$nb"
        "$PY" "$PATCH" "notebooks/$nb"
        if [ "$nb" = "06_product_definitions.ipynb" ]; then
            echo "executing notebook 06 (adds the product-scope chart output)..."
            "$JUPYTER" nbconvert --to notebook --execute --inplace \
                --ExecutePreprocessor.kernel_name=python3 "notebooks/$nb"
        fi
        git add notebooks
    fi
    if git diff --cached --quiet; then
        echo "nothing staged on $branch — nothing to do"
    else
        git commit -q -m "$msg

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
        git log --oneline -1
    fi
}

do_branch fmatch-4-tiling 04_tiling.ipynb \
    "Move tile-manager demo notebook to notebooks/ and add documentation links"
do_branch fmatch-5-weighting-aggregation 05_weighting_aggregation.ipynb \
    "Move weighting/aggregation demo notebook to notebooks/ and add documentation links"
do_branch fmatch-6-product-definitions 06_product_definitions.ipynb \
    "Move product-definitions demo notebook to notebooks/, add links and scope chart"
do_branch fmatch-7-product-assembly 07_product_assembly.ipynb \
    "Move product-assembly demo notebook to notebooks/ and add documentation links"
do_branch fmatch-8-camera-segmentation 08_camera_segmentation.ipynb \
    "Move camera-segmentation demo notebook to notebooks/ and add documentation links"
do_branch fmatch-9-runners 09_runners.ipynb \
    "Move runners demo notebook to notebooks/ and add documentation links"
do_branch scene-id-imager-family 10_scene_id_imager.ipynb \
    "Move Scene-ID imager demo notebook to notebooks/ and add documentation links"

echo ""
echo "=== Summary (branch : own notebook location : tip commit) ==="
for b in fmatch-1-geometry-psf fmatch-2-readers-gridded fmatch-3-readers-swath \
         fmatch-4-tiling fmatch-5-weighting-aggregation fmatch-6-product-definitions \
         fmatch-7-product-assembly fmatch-8-camera-segmentation fmatch-9-runners \
         scene-id-imager-family; do
    nb=$(git ls-tree -r --name-only "$b" | grep '^notebooks/' | tail -1 || true)
    echo "$b : ${nb:-MISSING} : $(git log --oneline -1 "$b" | head -c 80)"
done
echo ""
echo "Done. You are now on $(git branch --show-current)."
echo "This doc/notebook_move/ folder is untracked scaffolding — delete it when happy."
