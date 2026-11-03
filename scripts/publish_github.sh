#!/usr/bin/env bash
# Publish this independent project using the maintainer's authenticated GitHub CLI.
# Does not read or print access tokens. Requires explicit execution by the maintainer.
set -euo pipefail

OWNER="${GITHUB_OWNER:-h-yu-u}"
REPO="${GITHUB_REPO:-weightspace-doctor}"
VERSION="${VERSION:-v0.1.0}"
FULL_NAME="$OWNER/$REPO"
BASE_URL="https://github.com/$FULL_NAME"

command -v git >/dev/null || { echo 'git is required' >&2; exit 1; }
command -v gh >/dev/null || { echo 'Install GitHub CLI (brew install gh), then authenticate with gh auth login' >&2; exit 1; }
gh auth status >/dev/null || { echo 'Run gh auth login first' >&2; exit 1; }
[[ -d .git ]] || { echo 'Run from a Git checkout or restore with git clone <.bundle> first' >&2; exit 1; }
[[ "$(git branch --show-current)" == "main" ]] || { echo 'Switch to main before publishing' >&2; exit 1; }
[[ -z "$(git status --porcelain)" ]] || { echo 'Working tree is not clean; commit or stash changes before publishing' >&2; exit 1; }
git rev-parse --verify "refs/tags/$VERSION" >/dev/null || { echo "Missing tag: $VERSION" >&2; exit 1; }

if gh repo view "$FULL_NAME" >/dev/null 2>&1; then
    VISIBILITY="$(gh repo view "$FULL_NAME" --json visibility --jq .visibility)"
    [[ "$VISIBILITY" == "PUBLIC" ]] || { echo "Repository exists but is $VISIBILITY; refusing to publish an open-source release there." >&2; exit 1; }
    echo "Repository already exists: $BASE_URL"
else
    echo "Creating public GitHub repository: $FULL_NAME"
    gh repo create "$FULL_NAME" --public --description "LLM model diagnostics and reversible sparse weight-space repair" --disable-wiki
fi

if git remote get-url origin >/dev/null 2>&1; then
    REMOTE_URL="$(git remote get-url origin)"
    case "$REMOTE_URL" in
        "https://github.com/$FULL_NAME"|"https://github.com/$FULL_NAME.git"|"git@github.com:$FULL_NAME.git"|"git@github.com:$FULL_NAME") ;;
        *) echo "Existing origin $REMOTE_URL does not point to $FULL_NAME. Refusing to overwrite it." >&2; exit 1;;
    esac
else
    git remote add origin "https://github.com/$FULL_NAME.git"
fi

gh auth setup-git >/dev/null
git push -u origin main
git push origin "refs/tags/$VERSION"

if gh release view "$VERSION" --repo "$FULL_NAME" >/dev/null 2>&1; then
    echo "Release already exists: $BASE_URL/releases/tag/$VERSION"
else
    gh release create "$VERSION" --repo "$FULL_NAME" --title "$VERSION — Initial open-source preview" \
        --notes-file docs/RELEASE_NOTES_v0.1.0.md
fi

echo "Published repository: $BASE_URL"
echo "Published release: $BASE_URL/releases/tag/$VERSION"
