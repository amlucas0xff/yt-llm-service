# Test Frames

Place 10-20 representative PNG frames here for benchmarking.

## Recommended frame types:
1. Terminal/shell output (dark background, light text)
2. Code editor (VS Code, vim) with syntax highlighting
3. Presentation slides with bullet points
4. Browser content with navigation chrome
5. Mixed content (code + diagrams + text)

## How to extract frames from a YouTube video:

```bash
# Single frame at timestamp 5:30
ffmpeg -ss 5:30 -i video.mp4 -frames:v 1 terminal_dark.png

# Range of frames at 1fps
ffmpeg -ss 8:29 -to 9:15 -i video.mp4 -vf fps=1 slide_%04d.png
```

## Naming convention:
- `terminal_dark_01.png` — dark terminal
- `code_vscode_01.png` — VS Code editor
- `slide_text_01.png` — presentation slide
- `browser_article_01.png` — browser content
