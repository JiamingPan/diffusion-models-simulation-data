#!/usr/bin/env bash
set -euo pipefail

branch=${1:-}
repo_root=$(git rev-parse --show-toplevel)
cd "${repo_root}"

if [[ -z "${branch}" ]]; then
  branch=$(git branch --show-current)
fi
if [[ -z "${branch}" ]]; then
  echo "Could not infer current branch. Usage: bash scripts/gl_safe_pull.sh <branch>" >&2
  exit 2
fi

mapfile -t dirty_tracked < <(
  {
    git diff --name-only
    git diff --cached --name-only
  } | sort -u
)

dirty_notebooks=()
dirty_other=()
for path in "${dirty_tracked[@]}"; do
  [[ -z "${path}" ]] && continue
  case "${path}" in
    *.ipynb)
      dirty_notebooks+=("${path}")
      ;;
    results/figures/.gitkeep|results/tables/.gitkeep)
      # Great Lakes often has results/ as a symlink to Turbo storage. These
      # placeholder files can appear deleted locally and should not block pulls.
      ;;
    *)
      dirty_other+=("${path}")
      ;;
  esac
done

if [[ "${#dirty_other[@]}" -gt 0 ]]; then
  echo "Refusing to pull because non-notebook tracked files are dirty:" >&2
  printf '  %s\n' "${dirty_other[@]}" >&2
  echo "Commit, inspect, or manually back up those files first." >&2
  exit 1
fi

if [[ "${#dirty_notebooks[@]}" -gt 0 ]]; then
  backup_dir="backup_executed_notebooks_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "${backup_dir}"
  echo "Backing up executed notebooks to ${backup_dir}/"
  for path in "${dirty_notebooks[@]}"; do
    if [[ -e "${path}" ]]; then
      mkdir -p "${backup_dir}/$(dirname "${path}")"
      cp -p "${path}" "${backup_dir}/${path}"
    fi
  done
  git restore --staged --worktree -- "${dirty_notebooks[@]}"
else
  echo "No dirty tracked notebooks to back up."
fi

current_branch=$(git branch --show-current)
if [[ "${current_branch}" != "${branch}" ]]; then
  git switch "${branch}"
fi

git fetch origin "${branch}"

mapfile -t untracked_conflicts < <(
  comm -12 \
    <(git ls-files --others --exclude-standard | sort -u) \
    <(git ls-tree -r --name-only "origin/${branch}" | sort -u)
)

if [[ "${#untracked_conflicts[@]}" -gt 0 ]]; then
  backup_dir="backup_untracked_before_pull_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "${backup_dir}"
  echo "Backing up untracked files that would be overwritten to ${backup_dir}/"
  for path in "${untracked_conflicts[@]}"; do
    if [[ -e "${path}" ]]; then
      mkdir -p "${backup_dir}/$(dirname "${path}")"
      mv "${path}" "${backup_dir}/${path}"
      echo "  moved ${path}"
    fi
  done
fi

git pull --ff-only origin "${branch}"

echo "Safe pull complete:"
git log --oneline -1
