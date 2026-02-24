"""
Notes generation service using local gpt-oss-20b via llama.cpp HTTP API.

The llama-cpp sidecar exposes an OpenAI-compatible /v1/chat/completions endpoint.
gpt-oss models require the Harmony chat template, which is handled automatically
by llama-server's --jinja flag (the template is embedded in the GGUF).

IMPORTANT: generate() is async to avoid blocking FastAPI's event loop during
the long HTTP call to llama-cpp (model generation can take 30-300 seconds).
"""

import logging
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

CORRECTION_SYSTEM_PROMPT = """\
[ROLE]: Transcript correction specialist.
[TASK]: Fix speech-to-text errors in the primary transcript using the reference transcript and video metadata.

<ground_truth>
The Video Context block is authoritative ground truth.
- Title, channel, and tags contain correct spellings of all proper nouns in this video.
- Correct any word that is a phonetic approximation of a term in the Video Context to match exactly.
- This rule overrides all other rules. If both transcripts say "Entropic" but the title says "Anthropic", correct every instance to "Anthropic".
</ground_truth>

<correction_rules>
1. Fix words that are clearly misheard, phonetically substituted, or garbled.
2. Use the reference transcript to resolve ambiguous words — prefer the plausible technical term over a nonsense word.
3. Preserve sentence structure, punctuation style, speaker intent, and filler words exactly.
4. Output approximately the same length as the primary transcript.
5. Add nothing that appears in neither source.
</correction_rules>

<self_check>
Before outputting: scan the corrected transcript for any word phonetically resembling a proper noun from the Video Context. Correct any remaining mismatches.
</self_check>

<output_format>
Corrected transcript only. No preamble, labels, or explanation.
</output_format>
"""

CHUNK_WORDS = 2000  # words per correction chunk

TRUNCATION_NOTICE = (
    "\n\n[NOTE: Transcript was truncated due to length. "
    "The middle portion has been omitted. Analysis covers the beginning and end.]\n\n"
)


class NotesService:
    """Generates structured notes from a transcript using gpt-oss-20b via llama.cpp."""

    def __init__(self, config: Config):
        self.base_url = config.LLAMA_CPP_URL.rstrip("/")
        self.max_tokens = config.NOTES_MAX_TOKENS
        # connect=30s: llama-server should be up (Docker healthcheck enforces this)
        # read=300s: generation on a 20B model can be slow for long transcripts
        self.timeout = httpx.Timeout(connect=30.0, read=300.0, write=30.0, pool=30.0)

    def _truncate_transcript(self, text: str) -> str:
        """
        If transcript exceeds max_tokens (approximated as words * 1.3),
        preserve the first 25% and last 25%, truncating the middle.
        """
        approx_tokens = len(text.split()) * 1.3
        if approx_tokens <= self.max_tokens:
            return text

        logger.warning(
            f"Transcript too long (~{int(approx_tokens)} tokens). Truncating middle."
        )
        words = text.split()
        keep = int(len(words) * 0.25)
        first_part = " ".join(words[:keep])
        last_part = " ".join(words[-keep:])
        return first_part + TRUNCATION_NOTICE + last_part

    async def generate(
        self,
        transcript_text: str,
        video_context: Optional["VideoContext"] = None,
    ) -> Optional[str]:
        """
        Generate structured markdown notes from a transcript.

        This is async: it must be awaited from FastAPI async endpoints.
        Using AsyncClient avoids blocking the event loop during LLM generation.

        Args:
            transcript_text: Plain text transcript content.
            video_context: Optional metadata for proper noun grounding. When provided,
                           prepended to the user message as authoritative ground truth
                           so the model corrects misspellings even after GEC.

        Returns:
            Markdown string with structured notes, or None if generation fails.
        """
        if not transcript_text or not transcript_text.strip():
            logger.warning("Empty transcript passed to NotesService.generate()")
            return None

        log_action("Generating structured notes from transcript")

        transcript = self._truncate_transcript(transcript_text)

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
                return None

            content = choices[0].get("message", {}).get("content", "").strip()
            if not content:
                logger.error("llama-cpp returned empty content in choice[0]")
                return None

            logger.info(f"Notes generated successfully ({len(content)} chars)")
            return content

        except httpx.ConnectError as e:
            logger.warning(f"llama-cpp service unavailable: {e}")
            return None
        except httpx.TimeoutException as e:
            logger.warning(f"llama-cpp request timed out: {e}")
            return None
        except httpx.HTTPStatusError as e:
            logger.error(
                f"llama-cpp HTTP error {e.response.status_code}: {e.response.text[:200]}"
            )
            return None
        except Exception as e:
            logger.error(f"Unexpected error calling llama-cpp: {e}")
            return None

    def _build_context_block(self, video_context: "VideoContext") -> str:
        """Format VideoContext as an authoritative grounding block for LLM prompts.

        Positioned BEFORE transcript content so the model treats it as ground truth
        rather than supplementary context. Uses XML tags per GPT-5.2 prompting guide
        to signal authoritative scope to the model.
        """
        tags_str = ", ".join(video_context.tags) if video_context.tags else ""
        lines = ["<video_context>"]
        lines.append("Authoritative metadata — these spellings override both transcripts:")
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

    async def correct_transcript(
        self,
        whisperx_text: str,
        yt_captions_text: str,
        video_context: Optional["VideoContext"] = None,
    ) -> str:
        """
        Use gpt-oss-20b to correct the WhisperX transcript using YT captions as reference.

        For long transcripts, applies correction in ~2000-word chunks to avoid context
        limits. Always returns full-length text — never truncates. Falls back to the
        original whisperx_text on any error.
        """
        if not whisperx_text or not whisperx_text.strip():
            return whisperx_text
        if not yt_captions_text or not yt_captions_text.strip():
            return whisperx_text

        log_action("Running GEC transcript correction pass")

        wx_words = whisperx_text.split()

        # Short transcript: single-pass correction
        if len(wx_words) <= CHUNK_WORDS:
            return await self._correct_chunk(whisperx_text, yt_captions_text, video_context)

        # Long transcript: chunk-level correction
        # Split WhisperX into chunks; use a proportional window of YT captions per chunk
        yt_words = yt_captions_text.split()
        chunks = [wx_words[i:i + CHUNK_WORDS] for i in range(0, len(wx_words), CHUNK_WORDS)]
        corrected_chunks = []

        for i, chunk in enumerate(chunks):
            # Align a proportional YT captions window to this chunk
            frac_start = i / len(chunks)
            frac_end = (i + 1) / len(chunks)
            yt_start = int(frac_start * len(yt_words))
            yt_end = int(frac_end * len(yt_words))
            yt_chunk = " ".join(yt_words[yt_start:yt_end])
            wx_chunk = " ".join(chunk)

            corrected = await self._correct_chunk(wx_chunk, yt_chunk, video_context)
            corrected_chunks.append(corrected)

        result = " ".join(corrected_chunks)
        logger.info(f"GEC chunked correction complete ({len(chunks)} chunks, {len(result)} chars)")
        return result

    async def _correct_chunk(
        self,
        whisperx_chunk: str,
        yt_chunk: str,
        video_context: Optional["VideoContext"] = None,
    ) -> str:
        """Single-chunk correction call. Falls back to whisperx_chunk on error."""
        context_block = (
            self._build_context_block(video_context) + "\n\n"
            if video_context else ""
        )
        user_content = (
            f"{context_block}"
            "## WhisperX transcript (primary — correct this):\n"
            f"{whisperx_chunk}\n\n"
            "## YouTube auto-captions (reference — use to resolve ambiguous words):\n"
            f"{yt_chunk}"
        )
        payload = {
            "model": "gpt-oss-20b",
            "messages": [
                {"role": "developer", "content": CORRECTION_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.1,
            "max_tokens": 4000,
            "chat_template_kwargs": {"reasoning_effort": "low"},
        }
        url = f"{self.base_url}/v1/chat/completions"
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
            content = (
                response.json().get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
            return content if content else whisperx_chunk
        except Exception as e:
            logger.warning(f"GEC chunk correction failed (non-fatal): {e}")
            return whisperx_chunk
