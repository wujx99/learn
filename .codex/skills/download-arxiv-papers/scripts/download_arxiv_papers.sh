#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  download_arxiv_papers.sh [--repo-root PATH] [--jobs N] \
    --paper 'FOLDER=ARXIV_ID' [--paper 'FOLDER=ARXIV_ID' ...]

Downloads each arXiv source bundle to _inbox/papers/FOLDER/, extracts it
safely, and downloads the matching PDF to _inbox/pdfs/FOLDER.pdf.

Options:
  --repo-root PATH  Repository root. Defaults to the current Git root.
  --jobs N          Concurrent downloads, from 1 to 4. Default: 3.
  --paper SPEC      Folder label and arXiv ID separated by '='. Repeatable.
  -h, --help        Show this help.
USAGE
}

repo_root=""
jobs=3
declare -a requested_papers=()

while (($#)); do
  case "$1" in
    --repo-root)
      (($# >= 2)) || { echo "ERROR: --repo-root requires a path" >&2; exit 2; }
      repo_root="$2"
      shift 2
      ;;
    --jobs)
      (($# >= 2)) || { echo "ERROR: --jobs requires a number" >&2; exit 2; }
      jobs="$2"
      shift 2
      ;;
    --paper)
      (($# >= 2)) || { echo "ERROR: --paper requires FOLDER=ARXIV_ID" >&2; exit 2; }
      requested_papers+=("$2")
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

[[ "$jobs" =~ ^[1-4]$ ]] || { echo "ERROR: --jobs must be an integer from 1 to 4" >&2; exit 2; }
((${#requested_papers[@]} > 0)) || { echo "ERROR: provide at least one --paper" >&2; exit 2; }

if [[ -z "$repo_root" ]]; then
  repo_root=$(git rev-parse --show-toplevel 2>/dev/null) || {
    echo "ERROR: not inside a Git repository; pass --repo-root" >&2
    exit 2
  }
fi

repo_root=$(cd "$repo_root" && pwd -P)
source_root="$repo_root/_inbox/papers"
pdf_root="$repo_root/_inbox/pdfs"
mkdir -p "$source_root" "$pdf_root"

declare -A name_to_id=()
declare -A seen_ids=()
declare -a papers=()

for specification in "${requested_papers[@]}"; do
  [[ "$specification" == *=* ]] || {
    echo "ERROR: invalid paper specification '$specification'; expected FOLDER=ARXIV_ID" >&2
    exit 2
  }

  paper_name=${specification%%=*}
  arxiv_id=${specification#*=}

  [[ -n "$paper_name" && -n "$arxiv_id" ]] || {
    echo "ERROR: folder name and arXiv ID must both be non-empty" >&2
    exit 2
  }
  [[ "$paper_name" != "." && "$paper_name" != ".." && "$paper_name" != .* ]] || {
    echo "ERROR: unsafe folder name '$paper_name'" >&2
    exit 2
  }
  [[ "$paper_name" != *"/"* && "$paper_name" != *"\\"* && "$paper_name" != *$'\n'* && "$paper_name" != *$'\r'* ]] || {
    echo "ERROR: folder name contains a path separator or control character: '$paper_name'" >&2
    exit 2
  }
  [[ "$arxiv_id" =~ ^([0-9]{4}\.[0-9]{4,5}|[A-Za-z.-]+/[0-9]{7})(v[0-9]+)?$ ]] || {
    echo "ERROR: unsupported arXiv ID '$arxiv_id'" >&2
    exit 2
  }

  if [[ -n "${name_to_id[$paper_name]:-}" ]]; then
    if [[ "${name_to_id[$paper_name]}" == "$arxiv_id" ]]; then
      echo "DEDUPLICATED $paper_name ($arxiv_id)"
      continue
    fi
    echo "ERROR: folder '$paper_name' maps to both '${name_to_id[$paper_name]}' and '$arxiv_id'" >&2
    exit 2
  fi

  if [[ -n "${seen_ids[$arxiv_id]:-}" ]]; then
    echo "DEDUPLICATED $paper_name ($arxiv_id); already mapped to '${seen_ids[$arxiv_id]}'"
    continue
  fi

  name_to_id[$paper_name]="$arxiv_id"
  seen_ids[$arxiv_id]="$paper_name"
  papers+=("$paper_name=$arxiv_id")
done

download_one() (
  set -euo pipefail

  specification="$1"
  paper_name=${specification%%=*}
  arxiv_id=${specification#*=}
  final_source_dir="$source_root/$paper_name"
  final_pdf="$pdf_root/$paper_name.pdf"
  archive_name="arXiv-$arxiv_id-source.tar.gz"
  expected_archive="$final_source_dir/$archive_name"

  if [[ -e "$final_source_dir" || -e "$final_pdf" ]]; then
    if [[ -d "$final_source_dir" && -f "$expected_archive" && -f "$final_pdf" ]] \
      && gzip -t "$expected_archive" 2>/dev/null \
      && find "$final_source_dir" -type f -name '*.tex' -print -quit | grep -q . \
      && file -b "$final_pdf" | grep -q '^PDF document'; then
      echo "SKIPPED $paper_name ($arxiv_id): valid files already exist"
      exit 0
    fi
    echo "ERROR: refusing to overwrite incomplete or conflicting target for $paper_name" >&2
    exit 1
  fi

  stage_dir=$(mktemp -d "$source_root/.download-${paper_name}.XXXXXX")
  stage_pdf=$(mktemp "$pdf_root/.download-${paper_name}.XXXXXX.pdf")
  installed_pdf=0

  cleanup() {
    if [[ -n "${stage_pdf:-}" && -f "$stage_pdf" ]]; then
      rm -f -- "$stage_pdf"
    fi
    if [[ -n "${stage_dir:-}" && -d "$stage_dir" && "$stage_dir" == "$source_root"/.download-* ]]; then
      rm -rf -- "$stage_dir"
    fi
    if ((installed_pdf)) && [[ -f "$final_pdf" && ! -e "$final_source_dir" ]]; then
      rm -f -- "$final_pdf"
    fi
  }
  trap cleanup EXIT

  archive="$stage_dir/$archive_name"
  members="$stage_dir/.archive-members"
  listing="$stage_dir/.archive-listing"

  echo "DOWNLOADING $paper_name ($arxiv_id)"
  curl -fL --silent --show-error --retry 5 --retry-delay 3 --retry-connrefused \
    "https://arxiv.org/e-print/$arxiv_id" \
    -o "$archive"
  curl -fL --silent --show-error --retry 5 --retry-delay 3 --retry-connrefused \
    "https://arxiv.org/pdf/$arxiv_id" \
    -o "$stage_pdf"

  gzip -t "$archive" || { echo "ERROR: invalid gzip source for $paper_name" >&2; exit 1; }
  file -b "$stage_pdf" | grep -q '^PDF document' || {
    echo "ERROR: downloaded PDF is invalid for $paper_name" >&2
    exit 1
  }

  if tar -tzf "$archive" >"$members" 2>/dev/null; then
    if grep -Eq '(^/|^[A-Za-z]:[/\\]|(^|/)\.\.(/|$))' "$members"; then
      echo "ERROR: unsafe archive member in $paper_name" >&2
      exit 1
    fi
    tar -tvzf "$archive" >"$listing"
    if awk '$1 ~ /^[lh]/ { found=1 } END { exit found ? 0 : 1 }' "$listing"; then
      echo "ERROR: archive links are not allowed for $paper_name" >&2
      exit 1
    fi
    rm -f -- "$members" "$listing"
    tar -xzf "$archive" --no-same-owner --no-same-permissions -C "$stage_dir"
  else
    gzip -dc "$archive" >"$stage_dir/source.tex"
  fi

  find "$stage_dir" -type f -name '*.tex' -print -quit | grep -q . || {
    echo "ERROR: no TeX file found after extracting $paper_name" >&2
    exit 1
  }

  mv -- "$stage_pdf" "$final_pdf"
  stage_pdf=""
  installed_pdf=1
  mv -- "$stage_dir" "$final_source_dir"
  stage_dir=""
  installed_pdf=0
  trap - EXIT

  tex_count=$(find "$final_source_dir" -type f -name '*.tex' | wc -l)
  echo "COMPLETED $paper_name ($arxiv_id): $tex_count TeX file(s)"
)

export -f download_one
export source_root pdf_root

printf '%s\0' "${papers[@]}" |
  xargs -0 -n 1 -P "$jobs" bash -c 'download_one "$1"' _

echo "ALL COMPLETE: ${#papers[@]} unique paper(s)"
