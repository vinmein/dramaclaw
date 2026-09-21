from contextlib import asynccontextmanager

import pytest
from pydantic_ai.messages import ModelRequest, SystemPromptPart, UserPromptPart
from pydantic_ai.models.openai import OpenAIChatModel

from novelvideo import config
from novelvideo.utils.generation_language import (
    apply_generation_language,
    generation_language,
)
from novelvideo.utils.source_language import (
    resolve_asset_language,
    detect_asset_language,
)


@pytest.fixture(autouse=True)
def english_default(monkeypatch):
    monkeypatch.delenv("GENERATION_LANGUAGE", raising=False)


def test_english_generation_does_not_change_source_detection():
    assert generation_language() == "en"
    assert resolve_asset_language("她打开了门。") == "en"
    assert detect_asset_language("她打开了门。") == "zh"


def test_auto_mode_preserves_source_language(monkeypatch):
    monkeypatch.setenv("GENERATION_LANGUAGE", "auto")
    assert resolve_asset_language("她打开了门。") == "zh"
    assert resolve_asset_language("Maya opens the door.") == "en"


def test_policy_preserves_original_messages_and_machine_tokens():
    source = ModelRequest(
        parts=[SystemPromptPart("Use Chinese."), UserPromptPart("Make a short film.")]
    )
    messages = [source]
    result = apply_generation_language(messages)
    assert result is not messages
    assert len(source.parts) == 2
    assert result[0].parts[:2] == source.parts
    rule = result[0].parts[-1].content
    assert "English" in rule and "dialogue" in rule and "narration" in rule
    assert "Preserve JSON keys" in rule
    assert "spoken English" in rule


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_policy_reaches_regular_and_streaming_model_requests(
    monkeypatch, streaming
):
    seen = []

    async def request(self, messages, *args, **kwargs):
        seen.extend(messages)
        return "ok"

    @asynccontextmanager
    async def stream(self, messages, *args, **kwargs):
        seen.extend(messages)
        yield "ok"

    monkeypatch.setattr(OpenAIChatModel, "request", request)
    monkeypatch.setattr(OpenAIChatModel, "request_stream", stream)
    model = config._newapi_text_openai_model(
        "test",
        api_key="test",
        base_url="https://example.test/v1",
        timeout_seconds=1,
        profile=None,
    )
    messages = [ModelRequest(parts=[UserPromptPart("Create a script.")])]
    if streaming:
        async with model.request_stream(messages, None, None) as result:
            assert result == "ok"
    else:
        assert await model.request(messages, None, None) == "ok"
    assert "OUTPUT LANGUAGE POLICY" in seen[0].parts[-1].content
    assert len(messages[0].parts) == 1


@pytest.mark.asyncio
async def test_explicit_translation_model_is_exempt(monkeypatch):
    seen = {}
    monkeypatch.setattr(config, "is_ce_effective", lambda: True)
    monkeypatch.setattr(
        config,
        "get_newapi_runtime_credentials",
        lambda **kwargs: ("test", "https://example.test/v1"),
    )

    def model_factory(*args, **kwargs):
        seen.update(kwargs)
        return object()

    monkeypatch.setattr(config, "_newapi_text_openai_model", model_factory)
    config.get_newapi_text_pydantic_model("FREEZONE_TRANSLATION_MODEL", "translator")
    assert seen["output_language_policy"] is False
    config.get_newapi_text_pydantic_model("FREEZONE_STORY_SCRIPT_MODEL", "script")
    assert seen["output_language_policy"] is True


def test_script_and_vision_prompts_require_english():
    from novelvideo.freezone.text_node import (
        FREEZONE_STORY_SCRIPT_SYSTEM_PROMPT,
        FREEZONE_VIDEO_STORY_SCRIPT_SYSTEM_PROMPT,
        build_freezone_story_script_task,
    )

    for prompt in (
        FREEZONE_STORY_SCRIPT_SYSTEM_PROMPT,
        FREEZONE_VIDEO_STORY_SCRIPT_SYSTEM_PROMPT,
    ):
        assert "Write all generated prose in English" in prompt
        assert "Keep all fields in Simplified Chinese" not in prompt
        assert "spoken English" in prompt
    task = build_freezone_story_script_task(
        source_text="A woman finds a key.", prompt="A thriller"
    )
    assert "Write every generated prose field in English" in task
    assert "Use `None` when dialogue is absent." in task


def test_video_language_defaults_to_english_even_for_legacy_chinese_source():
    from novelvideo.seedance2_i2v.prompt import detect_seedance2_prompt_language
    from novelvideo.api.schemas import BeatVideoPromptGenerateRequest

    assert (
        detect_seedance2_prompt_language({"visual_description": "她打开了门。"}) == "en"
    )
    assert BeatVideoPromptGenerateRequest().language == "en"


def test_english_edge_voice_defaults():
    from novelvideo.generators.tts_generator import TTSParams, EdgeTTSGenerator

    assert TTSParams(text="Hello").voice.startswith("en-")
    assert EdgeTTSGenerator().voice.startswith("en-")


@pytest.mark.parametrize("builder", [
    "build_freezone_video_prompt", "build_freezone_image_to_video_prompt",
    "build_freezone_keyframe_video_prompt", "build_freezone_omni_video_prompt",
])
def test_canvas_video_modes_request_english_speech_without_forcing_audio(builder):
    from novelvideo.freezone import video_node
    prompt = getattr(video_node, builder)(user_prompt="A quiet sunrise.")
    assert "Any generated dialogue, narration or vocals must be in English" in prompt
    assert "Do not add speech" in prompt


def test_narration_guidance_is_english():
    from novelvideo.seedance2_i2v.voice_clone import narration_style_prompt, narration_style_label
    assert "English" in narration_style_prompt("first_person")
    assert "English" in narration_style_prompt("third_person")
    assert narration_style_label("first_person") == "First-person narration"
