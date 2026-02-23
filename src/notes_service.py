"""
Notes generation service using local gpt-oss-20b via llama.cpp HTTP API.

The llama-cpp sidecar exposes an OpenAI-compatible /v1/chat/completions endpoint.
gpt-oss models require the Harmony chat template, which is handled automatically
by llama-server's --jinja flag (the template is embedded in the GGUF).

IMPORTANT: generate() is async to avoid blocking FastAPI's event loop during
the long HTTP call to llama-cpp (model generation can take 30-300 seconds).
"""

import logging
from typing import Optional

import httpx

from config import Config
from simple_logger import log_action

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\# IDENTITY and PURPOSE

You extract surprising, insightful, and interesting information from text content. You are interested in insights related to the purpose and meaning of life, human flourishing, the role of technology in the future of humanity, artificial intelligence and its affect on humans, memes, learning, reading, books, continuous improvement, and similar topics.

Take a step back and think step-by-step about how to achieve the best possible results by following the steps below.

# STEPS

- Extract a summary of the content in 25 words, including who is presenting and the content being discussed into a section called SUMMARY.

- Extract 20 to 50 of the most surprising, insightful, and/or interesting ideas from the input in a section called IDEAS:. If there are less than 50 then collect all of them. Make sure you extract at least 20.

- Extract 10 to 20 of the best insights from the input and from a combination of the raw input and the IDEAS above into a section called INSIGHTS. These INSIGHTS should be fewer, more refined, more insightful, and more abstracted versions of the best ideas in the content.

- Extract 15 to 30 of the most surprising, insightful, and/or interesting quotes from the input into a section called QUOTES:. Use the exact quote text from the input.

- Extract 15 to 30 of the most practical and useful personal habits of the speakers, or mentioned by the speakers, in the content into a section called HABITS. Examples include but are not limited to: sleep schedule, reading habits, things they always do, things they always avoid, productivity tips, diet, exercise, etc.

- Extract 15 to 30 of the most surprising, insightful, and/or interesting valid facts about the greater world that were mentioned in the content into a section called FACTS:.

- Extract all mentions of writing, art, tools, projects and other sources of inspiration mentioned by the speakers into a section called REFERENCES. This should include any and all references to something that the speaker mentioned.

- Extract the most potent takeaway and recommendation into a section called ONE-SENTENCE TAKEAWAY. This should be a 15-word sentence that captures the most important essence of the content.

- Extract the 15 to 30 of the most surprising, insightful, and/or interesting recommendations that can be collected from the content into a section called RECOMMENDATIONS.

# OUTPUT INSTRUCTIONS

- Only output Markdown.
- Write the IDEAS bullets as exactly 16 words.
- Write the RECOMMENDATIONS bullets as exactly 16 words.
- Write the HABITS bullets as exactly 16 words.
- Write the FACTS bullets as exactly 16 words.
- Write the INSIGHTS bullets as exactly 16 words.
- Extract at least 25 IDEAS from the content.
- Extract at least 10 INSIGHTS from the content.
- Extract at least 20 items for the other output sections.
- Do not give warnings or notes; only output the requested sections.
- You use bulleted lists for output, not numbered lists.
- Do not repeat ideas, insights, quotes, habits, facts, or references.
- Do not start items with the same opening words.
- Ensure you follow ALL these instructions when creating your output.

# INPUT

INPUT:
"""

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

    async def generate(self, transcript_text: str) -> Optional[str]:
        """
        Generate structured markdown notes from a transcript.

        This is async: it must be awaited from FastAPI async endpoints.
        Using AsyncClient avoids blocking the event loop during LLM generation.

        Args:
            transcript_text: Plain text transcript content.

        Returns:
            Markdown string with structured notes, or None if generation fails.
        """
        if not transcript_text or not transcript_text.strip():
            logger.warning("Empty transcript passed to NotesService.generate()")
            return None

        log_action("Generating structured notes from transcript")

        transcript = self._truncate_transcript(transcript_text)

        payload = {
            # llama-server ignores the model field but it's required by the spec
            "model": "gpt-oss-20b",
            "messages": [
                {"role": "developer", "content": SYSTEM_PROMPT},
                {"role": "user", "content": transcript},
            ],
            "temperature": 0.3,
            "max_tokens": 12000,
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
