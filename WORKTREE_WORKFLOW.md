# Git Worktree Workflow

This project uses git worktrees to maintain separation between production-ready code and beta features under development.

## Overview

**Master Worktree** (Production)
- Location: `/home/amlucas/Desktop/yt-llm-service`
- Branch: `master`
- Purpose: Production-ready, stable code only
- Status: Clean, deployable at any time

**Feature Worktree** (Code Extraction Beta)
- Location: `/home/amlucas/Desktop/yt-llm-service-worktrees/code-extraction`
- Branch: `feature/code-extraction`
- Purpose: Code Extraction Suite development
- Status: Beta, under active development

## Working with Worktrees

### List All Worktrees

```bash
git worktree list
```

Output:
```
/home/amlucas/Desktop/yt-llm-service                            [master]
/home/amlucas/Desktop/yt-llm-service-worktrees/code-extraction  [feature/code-extraction]
```

### Switching Between Worktrees

Simply change directories:

```bash
# Work on production code
cd /home/amlucas/Desktop/yt-llm-service

# Work on Code Extraction feature
cd /home/amlucas/Desktop/yt-llm-service-worktrees/code-extraction
```

No need for `git stash` or `git checkout` - each worktree maintains its own working directory state.

### Making Changes

#### In Master Worktree (Production)
```bash
cd /home/amlucas/Desktop/yt-llm-service

# Make production fixes/improvements
git add .
git commit -m "fix: production bug fix"
git push origin master
```

#### In Feature Worktree (Code Extraction)
```bash
cd /home/amlucas/Desktop/yt-llm-service-worktrees/code-extraction

# Work on Code Extraction features
git add .
git commit -m "feat: improve OCR accuracy"
git push origin feature/code-extraction
```

### Docker Development

Each worktree can run its own Docker container:

#### Master Worktree
```bash
cd /home/amlucas/Desktop/yt-llm-service
docker-compose up -d
# Runs production service on port 8002
```

#### Feature Worktree
```bash
cd /home/amlucas/Desktop/yt-llm-service-worktrees/code-extraction

# Optionally modify docker-compose.yml port to avoid conflicts
# Change port 8002 to 8003 if running both simultaneously
docker-compose up -d
```

### Syncing Changes

If you need changes from master in your feature branch:

```bash
cd /home/amlucas/Desktop/yt-llm-service-worktrees/code-extraction
git fetch origin
git merge origin/master
# Resolve any conflicts
git commit
```

### Merging Feature to Master

When Code Extraction Suite is production-ready:

```bash
# 1. Ensure feature branch is up to date
cd /home/amlucas/Desktop/yt-llm-service-worktrees/code-extraction
git fetch origin
git merge origin/master
git push origin feature/code-extraction

# 2. Switch to master worktree
cd /home/amlucas/Desktop/yt-llm-service

# 3. Create merge commit or pull request
git merge feature/code-extraction
# Or use GitHub UI to create PR from feature/code-extraction to master

# 4. Test thoroughly in master
# 5. Push to production
git push origin master
```

### Removing Feature Worktree

Once feature is merged and no longer needed:

```bash
# 1. Remove worktree
git worktree remove /home/amlucas/Desktop/yt-llm-service-worktrees/code-extraction

# 2. Delete local branch
git branch -d feature/code-extraction

# 3. Delete remote branch (optional)
git push origin --delete feature/code-extraction
```

## Current Project Structure

```
/home/amlucas/Desktop/
├── yt-llm-service/                      # Master worktree (production)
│   ├── src/
│   │   ├── run_llm_api.py
│   │   ├── transcription_service.py
│   │   ├── audio_downloader.py
│   │   └── ... (production code only)
│   ├── docker-compose.yml               # Production config
│   ├── requirements.txt                 # Production dependencies
│   └── README.md                        # Production documentation
│
└── yt-llm-service-worktrees/
    └── code-extraction/                 # Feature worktree (beta)
        ├── src/
        │   ├── run_llm_api.py           # Same as master
        │   ├── code_detector.py         # Code Extraction specific
        │   ├── code_ocr_processor.py    # Code Extraction specific
        │   ├── code_consolidator.py     # Code Extraction specific
        │   ├── frame_extractor.py       # Code Extraction specific
        │   └── code_extraction_service.py
        ├── docs/
        │   ├── CODE_EXTRACTION.md
        │   └── 20250930120806-CodeExtractionImplementation.md
        ├── tests/poc/
        ├── extract_youtube_sample.py
        ├── start.sh
        ├── docker-compose.yml           # With Code Extraction volumes
        ├── requirements.txt             # With OCR dependencies
        └── README.md                    # With Code Extraction docs
```

## Best Practices

### Do's
- Make all Code Extraction changes in the feature worktree
- Keep master worktree clean and production-ready
- Commit frequently in feature worktree
- Test thoroughly before merging to master
- Use descriptive commit messages
- Document significant changes

### Don'ts
- Don't add Code Extraction code to master worktree
- Don't delete worktree while uncommitted changes exist
- Don't push experimental code to master
- Don't run conflicting Docker containers on the same port
- Don't forget to sync feature branch with master regularly

## Advantages of This Approach

1. **Isolation**: Beta features completely separated from production
2. **Parallel Work**: Work on production fixes while developing features
3. **No Context Switching**: No need for stashing or branch switching
4. **Clean History**: Clear separation of concerns in git history
5. **Safe Testing**: Test beta features without risking production
6. **Easy Rollback**: Simple to abandon features without affecting master
7. **Multiple Environments**: Run production and beta Docker containers simultaneously

## Troubleshooting

### Worktree Not Found
```bash
# List all worktrees
git worktree list

# Re-add if accidentally removed
git worktree add /home/amlucas/Desktop/yt-llm-service-worktrees/code-extraction feature/code-extraction
```

### Conflicting Docker Ports
```bash
# Change port in feature worktree's docker-compose.yml
ports:
  - "8003:8002"  # External:Internal
```

### Sync Issues
```bash
# In feature worktree
git fetch origin
git status
git merge origin/master
```

## References

- [Git Worktree Documentation](https://git-scm.com/docs/git-worktree)
- [Project README](README.md)
- [Code Extraction Documentation](https://github.com/yourusername/yt-llm-service/tree/feature/code-extraction/docs/CODE_EXTRACTION.md)

## Questions?

For questions about this workflow or the Code Extraction Suite:
- Check feature branch README: `/home/amlucas/Desktop/yt-llm-service-worktrees/code-extraction/README.md`
- Review Code Extraction docs: `/home/amlucas/Desktop/yt-llm-service-worktrees/code-extraction/docs/CODE_EXTRACTION.md`
- Open an issue on GitHub
