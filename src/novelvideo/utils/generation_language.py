"""Deployment-wide language for newly generated creative content."""

from __future__ import annotations

import os


def generation_language() -> str:
    """English by default; ``auto`` retains source-language behavior."""
    value = os.environ.get("GENERATION_LANGUAGE", "en").strip().lower()
    return value if value in {"en", "zh", "auto"} else "en"


def generation_language_instruction() -> str:
    language = generation_language()
    if language == "auto":
        return ""
    target = "English" if language == "en" else "Simplified Chinese"
    return (
        f"OUTPUT LANGUAGE POLICY: Write all newly generated user-visible content in {target}, "
        "including titles, scripts, shot descriptions, dialogue, narration, subtitles, "
        "image prompts, video motion prompts, sound descriptions and explanations. "
        f"For video with generated speech, explicitly request spoken {target}. "
        "This output-language policy overrides language defaults and examples in other prompts. "
        "Preserve JSON keys, required enum values, IDs, asset references, URLs, numbers and "
        "supplied proper names exactly. Do not translate machine-readable protocol tokens. "
        "For extraction or verbatim-source fields, preserve the original source text; "
        f"write newly authored dialogue and narration in {target}. "
        "Do not insert a language instruction into text that will itself be read aloud."
    )


def apply_generation_language(messages):
    """Add policy to a request copy without changing stored conversation history."""
    from dataclasses import replace
    from pydantic_ai.messages import ModelRequest, SystemPromptPart

    instruction = generation_language_instruction()
    if not instruction:
        return messages
    copied = list(messages)
    # Keep the last author request and tool-response ordering intact. The policy
    # is a system part, never speech text or a new user message.
    for index in range(len(copied) - 1, -1, -1):
        message = copied[index]
        if isinstance(message, ModelRequest):
            copied[index] = replace(
                message, parts=[*message.parts, SystemPromptPart(instruction)]
            )
            return copied
    return [ModelRequest(parts=[SystemPromptPart(instruction)]), *copied]


def media_language_instruction() -> str:
    """Constrain generated speech/text without adding speech to silent shots."""
    language = generation_language()
    if language == "auto":
        return ""
    target = "English" if language == "en" else "Simplified Chinese"
    return (
        f"Language: Any generated dialogue, narration or vocals must be in {target}. "
        f"If on-screen text or subtitles are explicitly requested, write them in {target}. "
        "Do not add speech, lyrics, subtitles or text when they were not requested. "
        "Preserve supplied names and asset identifiers."
    )
