#!/bin/bash

# Script to sync fork with upstream repository
# Usage: ./sync-fork.sh

set -e

echo "🔄 Syncing fork with upstream..."

# Fetch the latest changes from upstream
echo "📥 Fetching upstream changes..."
git fetch upstream

# Get current branch
CURRENT_BRANCH=$(git branch --show-current)
echo "📍 Current branch: $CURRENT_BRANCH"

# Switch to main branch (or master if main doesn't exist)
if git show-ref --verify --quiet refs/heads/main; then
    DEFAULT_BRANCH="main"
elif git show-ref --verify --quiet refs/heads/master; then
    DEFAULT_BRANCH="master"
else
    echo "❌ Could not find main or master branch"
    exit 1
fi

echo "🔀 Switching to $DEFAULT_BRANCH branch..."
git checkout $DEFAULT_BRANCH

# Merge upstream changes
echo "🔄 Merging upstream/$DEFAULT_BRANCH..."
git merge upstream/$DEFAULT_BRANCH

# Push changes to origin
echo "⬆️  Pushing changes to origin..."
git push origin $DEFAULT_BRANCH

# Switch back to original branch if it wasn't the default branch
if [ "$CURRENT_BRANCH" != "$DEFAULT_BRANCH" ]; then
    echo "🔀 Switching back to $CURRENT_BRANCH..."
    git checkout $CURRENT_BRANCH
fi

echo "✅ Fork sync completed!"
