#!/usr/bin/env bash
# Install the Magpie skill into Cursor, Claude Code, or Codex.
# Usage: ./install-skill.sh [cursor|claude|codex|all] [--global|--project [DIR]]
#   Default: --global (install to user home). Use --project to install into current or given directory.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAGPIE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SKILL_SRC="$MAGPIE_ROOT/skills/magpie"

if [[ ! -f "$SKILL_SRC/SKILL.md" ]]; then
  echo "Error: Skill not found at $SKILL_SRC (no SKILL.md)" >&2
  exit 1
fi

usage() {
  echo "Usage: $0 [cursor|claude|codex|all] [--global|--project [DIR]]"
  echo "  cursor  - Install for Cursor (~/.cursor/skills/magpie or project .cursor/skills/magpie)"
  echo "  claude  - Install for Claude Code (~/.claude/skills/magpie or project .claude/skills/magpie)"
  echo "  codex   - Install for Codex (~/.codex/skills/magpie or project .codex/skills/magpie)"
  echo "  all     - Install for all of the above"
  echo "  --global   - Install to user home (default)"
  echo "  --project  - Install to current directory (or DIR if given) as project scope"
  exit 1
}

TARGET=""
SCOPE="global"
PROJECT_DIR="."

while [[ $# -gt 0 ]]; do
  case "$1" in
    cursor|claude|codex|all) TARGET="$1" ;;
    --global)  SCOPE="global" ;;
    --project) SCOPE="project"; [[ -n "${2:-}" && "${2:0:1}" != - ]] && PROJECT_DIR="$2" && shift ;;
    -h|--help) usage ;;
    *) echo "Unknown option: $1" >&2; usage ;;
  esac
  shift
done

if [[ -z "$TARGET" ]]; then
  usage
fi

install_to() {
  local dest="$1"
  local name="$2"
  mkdir -p "$(dirname "$dest")"
  if [[ -d "$dest" ]]; then
    rm -rf "$dest"
  fi
  cp -r "$SKILL_SRC" "$dest"
  echo "Installed Magpie skill for $name at $dest"
}

if [[ "$SCOPE" == "global" ]]; then
  case "$TARGET" in
    cursor)
      install_to "$HOME/.cursor/skills/magpie" "Cursor (global)"
      ;;
    claude)
      install_to "$HOME/.claude/skills/magpie" "Claude Code (global)"
      ;;
    codex)
      install_to "$HOME/.codex/skills/magpie" "Codex (global)"
      ;;
    all)
      install_to "$HOME/.cursor/skills/magpie" "Cursor (global)"
      install_to "$HOME/.claude/skills/magpie" "Claude Code (global)"
      install_to "$HOME/.codex/skills/magpie" "Codex (global)"
      ;;
  esac
else
  PROJECT_DIR="$(cd "$PROJECT_DIR" && pwd)"
  case "$TARGET" in
    cursor)
      install_to "$PROJECT_DIR/.cursor/skills/magpie" "Cursor (project)"
      ;;
    claude)
      install_to "$PROJECT_DIR/.claude/skills/magpie" "Claude Code (project)"
      ;;
    codex)
      install_to "$PROJECT_DIR/.codex/skills/magpie" "Codex (project)"
      ;;
    all)
      install_to "$PROJECT_DIR/.cursor/skills/magpie" "Cursor (project)"
      install_to "$PROJECT_DIR/.claude/skills/magpie" "Claude Code (project)"
      install_to "$PROJECT_DIR/.codex/skills/magpie" "Codex (project)"
      ;;
  esac
fi
