"""
Notes generation service using local gpt-oss-20b via llama.cpp HTTP API.

The llama-cpp sidecar exposes an OpenAI-compatible /v1/chat/completions endpoint.
gpt-oss models require the Harmony chat template, which is handled automatically
by llama-server's --jinja flag (the template is embedded in the GGUF).

IMPORTANT: generate() is async to avoid blocking FastAPI's event loop during
the long HTTP call to llama-cpp (model generation can take 30-300 seconds).
"""

import logging
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from audio_downloader import VideoContext

from config import Config
from simple_logger import log_action

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
[ROLE]: Expert knowledge synthesizer and technical writer.
[AUDIENCE]: Engineers and knowledge workers who want a complete reference from a video without rewatching it.

[TASK]: Convert the transcript into a comprehensive structured notes document.

[STEPS]:
1. Infer a descriptive title from the content.
2. Write an Overview (3-5 sentences): what is covered, the core argument, and the conclusion.
3. Identify all major topics discussed. For each topic, create a section with:
   - A 2-4 sentence narrative explaining the concept or argument in full.
   - Key terms, tools, or names introduced — list each with a one-sentence definition.
   - One verbatim quote that best captures the speaker's point on this topic.
4. Write a Takeaways section with 4-8 concrete, actionable bullet points a practitioner can apply immediately.
5. Write a References section: list every tool, library, project, person, and external resource mentioned. Format each as: `name — one-sentence description of what it is or why it was mentioned`.

[CONSTRAINTS]:
- Preserve ALL important technical details, names, tools, and concepts from the transcript.
- Do not compress or omit topics for brevity — depth is the goal.
- Each topic section must be self-contained and informative without reading the transcript.
- Output only the markdown document. No preamble, no explanation.

[OUTPUT FORMAT]:
# <Title>

## Overview
<paragraph>

## <Topic 1>
<narrative paragraph>

**Key concepts:** term — definition; term — definition

> "<verbatim quote>"

## <Topic N>
...

## Takeaways
- <actionable point>

## References
- `tool/person/project` — <what it is or why mentioned>
"""

TRUNCATION_NOTICE = (
    "\n\n[NOTE: Transcript was truncated due to length. "
    "The middle portion has been omitted. Analysis covers the beginning and end.]\n\n"
)


@dataclass
class NotesResult:
    """Generated notes plus how faithfully the transcript reached the model.

    `truncated` is provenance, not an error: the notes are still usable, but
    they only cover the head and tail of the transcript. It is written into the
    Obsidian frontmatter so a reader can tell coverage from completeness.
    """

    text: Optional[str] = None
    truncated: bool = False


class NotesService:
    """Generates structured notes from a transcript using gpt-oss-20b via llama.cpp."""

    def __init__(self, config: Config):
        self.base_url = config.LLAMA_CPP_URL.rstrip("/")
        self.max_tokens = config.NOTES_MAX_TOKENS
        # connect=30s: llama-server should be up (Docker healthcheck enforces this)
        # read=300s: generation on a 20B model can be slow for long transcripts
        self.timeout = httpx.Timeout(connect=30.0, read=300.0, write=30.0, pool=30.0)

    def _truncate_transcript(self, text: str) -> tuple[str, bool]:
        """
        If transcript exceeds max_tokens (approximated as words * 1.3),
        preserve the first 25% and last 25%, truncating the middle.

        Returns (transcript, truncated).
        """
        approx_tokens = len(text.split()) * 1.3
        if approx_tokens <= self.max_tokens:
            return text, False

        logger.warning(
            f"Transcript too long (~{int(approx_tokens)} tokens). Truncating middle."
        )
        words = text.split()
        keep = int(len(words) * 0.25)
        first_part = " ".join(words[:keep])
        last_part = " ".join(words[-keep:])
        return first_part + TRUNCATION_NOTICE + last_part, True

    async def generate(
        self,
        transcript_text: str,
        video_context: Optional["VideoContext"] = None,
    ) -> NotesResult:
        """
        Generate structured markdown notes from a transcript.

        This is async: it must be awaited from FastAPI async endpoints.
        Using AsyncClient avoids blocking the event loop during LLM generation.

        Args:
            transcript_text: Plain text transcript content.
            video_context: Optional metadata for proper noun grounding. When provided,
                           prepended to the user message as authoritative ground truth
                           so the model spells names the way the video does.

        Returns:
            NotesResult. `text` is None if generation fails; `truncated` says
            whether the transcript's middle was dropped before prompting.
        """
        if not transcript_text or not transcript_text.strip():
            logger.warning("Empty transcript passed to NotesService.generate()")
            return NotesResult()

        log_action("Generating structured notes from transcript")

        transcript, truncated = self._truncate_transcript(transcript_text)

        if video_context:
            user_content = (
                self._build_context_block(video_context) + "\n\n"
                "<transcript>\n"
                + transcript
                + "\n</transcript>"
            )
        else:
            user_content = transcript

        payload = {
            # llama-server ignores the model field but it's required by the spec
            "model": "gpt-oss-20b",
            "messages": [
                {"role": "developer", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.3,
            "max_tokens": 12000,
            # Harmony chat template: set reasoning_effort to low so the model
            # spends minimal tokens on chain-of-thought and more on output.
            # The --jinja flag in llama-server enables this from the GGUF.
            "chat_template_kwargs": {"reasoning_effort": "medium"},
        }

        url = f"{self.base_url}/v1/chat/completions"
        logger.info(f"Calling llama-cpp at {url}")

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()

            data = response.json()
            choices = data.get("choices", [])
            if not choices:
                logger.error("llama-cpp returned empty choices list")
                return NotesResult(truncated=truncated)

            content = choices[0].get("message", {}).get("content", "").strip()
            if not content:
                logger.error("llama-cpp returned empty content in choice[0]")
                return NotesResult(truncated=truncated)

            logger.info(f"Notes generated successfully ({len(content)} chars)")
            return NotesResult(text=content, truncated=truncated)

        except httpx.ConnectError as e:
            logger.warning(f"llama-cpp service unavailable: {e}")
            return NotesResult(truncated=truncated)
        except httpx.TimeoutException as e:
            logger.warning(f"llama-cpp request timed out: {e}")
            return NotesResult(truncated=truncated)
        except httpx.HTTPStatusError as e:
            logger.error(
                f"llama-cpp HTTP error {e.response.status_code}: {e.response.text[:200]}"
            )
            return NotesResult(truncated=truncated)
        except Exception as e:
            logger.error(f"Unexpected error calling llama-cpp: {e}")
            return NotesResult(truncated=truncated)

    def _build_context_block(self, video_context: "VideoContext") -> str:
        """Format VideoContext as an authoritative grounding block for LLM prompts.

        Positioned BEFORE transcript content so the model treats it as ground truth
        rather than supplementary context. Uses XML tags per GPT-5.2 prompting guide
        to signal authoritative scope to the model.
        """
        tags_str = ", ".join(video_context.tags) if video_context.tags else ""
        lines = ["<video_context>"]
        lines.append("Authoritative metadata — these spellings override the transcript:")
        if video_context.title:
            lines.append(f"Title: {video_context.title}")
        if video_context.channel:
            lines.append(f"Channel: {video_context.channel}")
        if tags_str:
            lines.append(f"Tags: {tags_str}")
        if video_context.description:
            lines.append(f"Description: {video_context.description}")
        lines.append("</video_context>")
        return "\n".join(lines)
