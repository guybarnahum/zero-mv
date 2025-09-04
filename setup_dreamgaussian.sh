#!/usr/bin/env bash
set -euo pipefail

# Installs DreamGaussian’s CUDA extensions without interactive git prompts.
# It prefers unauthenticated ZIP downloads from codeload.github.com and falls
# back to plain HTTPS git clones if needed.

# Repos & branches (both default to main)
SIMPLE_KNN_REPO="ashawkey/simple-knn"
DGR_REPO="ashawkey/diff-gaussian-rasterization"
BRANCH="${1:-main}"

TMPDIR="$(mktemp -d)"
cleanup() { rm -rf "$TMPDIR"; }
trap cleanup EXIT

download_and_install_zip () {
  local repo="$1"
  local pkgname="$2"
  local branch="$3"

  local zip_url="https://codeload.github.com/${repo}/zip/refs/heads/${branch}"
  local zip_file="${TMPDIR}/${pkgname}.zip"
  local extract_dir="${TMPDIR}/${pkgname}"

  echo "→ Installing ${pkgname} via ZIP from ${zip_url}"
  if curl -fL "$zip_url" -o "$zip_file"; then
    mkdir -p "$extract_dir"
    unzip -q "$zip_file" -d "$extract_dir"
    # The extracted folder is REPO-BRANCH (e.g., simple-knn-main)
    local inner_dir
    inner_dir="$(find "$extract_dir" -maxdepth 1 -type d -name "*-${branch}" -print -quit)"
    if [[ -z "${inner_dir:-}" ]]; then
      echo "  ✖ Could not locate extracted folder for ${pkgname}"
      return 1
    fi
    python -m pip install --no-input "$inner_dir"
    echo "  ✓ ${pkgname} installed from ZIP"
    return 0
  fi
  echo "  ✖ ZIP download failed for ${pkgname}"
  return 1
}

fallback_git_install () {
  local repo="$1"
  local pkgname="$2"
  local branch="$3"

  echo "→ ZIP failed; trying git clone for ${pkgname} …"
  GIT_TERMINAL_PROMPT=0 GIT_ASKPASS= \
  git -c credential.helper= \
      -c url."https://github.com/".insteadof=git@github.com: \
      -c url."https://github.com/".insteadof=ssh://git@github.com/ \
      -c url."https://".insteadof=git:// \
      clone --depth=1 --branch "$branch" "https://github.com/${repo}.git" "${TMPDIR}/${pkgname}"

  python -m pip install --no-input "${TMPDIR}/${pkgname}"
  echo "  ✓ ${pkgname} installed via git clone"
}

echo "Checking nvcc for CUDA extension build…"
if ! command -v nvcc >/dev/null 2>&1; then
  echo "✖ 'nvcc' not found. Install CUDA toolkit and ensure nvcc is on PATH (needed to build extensions)."
  exit 1
fi

echo "Compiling DreamGaussian CUDA extensions (branch: ${BRANCH})"

# --- simple-knn ---
if ! download_and_install_zip "$SIMPLE_KNN_REPO" "simple-knn" "$BRANCH"; then
  fallback_git_install "$SIMPLE_KNN_REPO" "simple-knn" "$BRANCH"
fi

# --- diff-gaussian-rasterization ---
if ! download_and_install_zip "$DGR_REPO" "diff-gaussian-rasterization" "$BRANCH"; then
  fallback_git_install "$DGR_REPO" "diff-gaussian-rasterization" "$BRANCH"
fi

echo "✓ DreamGaussian CUDA deps installed."
