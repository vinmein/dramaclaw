"""Freezone 文本工具辅助逻辑。

当前包含：
- 中英文提示词互译
- 自由文本生成
- 故事脚本生成
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from novelvideo.egress_context import TrustedEgressContext
from novelvideo.model_gateway_runtime import (
    current_model_gateway_context,
    model_gateway_output_retries,
)

from novelvideo.official_defaults import (
    DEFAULT_FREEZONE_STORY_SCRIPT_MODEL,
    DEFAULT_FREEZONE_TEXT_WRITER_MODEL,
    DEFAULT_FREEZONE_TRANSLATION_MODEL,
)

FREEZONE_TRANSLATION_PROVIDER = "newapi"
FREEZONE_TRANSLATION_MODEL = DEFAULT_FREEZONE_TRANSLATION_MODEL
FREEZONE_TEXT_WRITER_MODEL = DEFAULT_FREEZONE_TEXT_WRITER_MODEL
FREEZONE_STORY_SCRIPT_MODEL = {
    "id": DEFAULT_FREEZONE_STORY_SCRIPT_MODEL,
    "provider": "newapi",
    "model": DEFAULT_FREEZONE_STORY_SCRIPT_MODEL,
    "label": "DramaClawAPI Story Script",
}
LEGACY_FREEZONE_STORY_SCRIPT_MODEL_IDS = {
    "newapi_gemini_flash",
    "openrouter_gemini_flash",
    "OpenRouter Gemini 2.5 Flash",
}

FREEZONE_TRANSLATION_SYSTEM_PROMPT = """# Freezone Prompt Translator

You translate prompting text between Simplified Chinese and English for creative nodes.

## Goal
- First determine the dominant natural language of the source text.
- If the dominant natural language is English, translate natural-language content into Simplified Chinese.
- If the dominant natural language is Simplified Chinese, translate natural-language content into English.
- Translate accurately while preserving prompting intent.
- Keep the output concise, directly usable as a prompt.
- Preserve cinematic, visual, audio, and motion terminology naturally.

## Rules
1. For mixed-language prompts, use the dominant natural language to decide the opposite target language.
2. Translate all natural-language content that should be user-readable into the target language.
3. Preserve line breaks, list structure, tags, and prompt segmentation when possible.
4. Keep IDs, asset markers, variable names, file names, model names, color codes, bracket tags, and technical tokens intact.
   Examples: [CM_6932], [YZSZ_974d], #00FFFF, 16:9, v2.0, fal.ai.
5. Do not add new details not present in the source.
6. For image/video/audio/text prompting, prefer natural creator-facing wording over literal textbook translation.
7. Only return the source directly when the detected source_language is exactly the same as the target_language.
8. If source_language and target_language differ, copying the original prose is a failure.
9. When translating English into Chinese, keep technical tokens intact but translate every English instruction sentence, rule sentence, heading, and description into Simplified Chinese.
10. When translating Chinese into English, keep technical tokens intact but translate every Chinese instruction sentence, rule sentence, heading, and description into English.
11. Return structured data matching the requested schema. Do not wrap with markdown.
"""

FREEZONE_TEXT_WRITER_SYSTEM_PROMPT = """# Freezone AI Text Writer

You create polished, creator-ready text from a user's instruction.

## Goal
- Write the requested story, scene, character setting, dialogue, outline, or creative prompt.
- Write stories, dialogue, narration and creative prompts in English. Follow the requested structure, tone, length and formatting.
- Make the result concrete and directly usable in a creative workflow.

## Rules
1. Preserve names, IDs, technical tokens, ratios, and other constraints supplied by the user.
2. Do not invent constraints that conflict with the user's instruction.
3. Return only the finished text. Do not explain your process.
4. Do not wrap the result in a markdown code fence.
"""

FREEZONE_STORY_SCRIPT_SYSTEM_PROMPT = """# Freezone Story Script Generator

Create a production-ready story-script table from the supplied script, idea or reference.
Write all generated prose in English, including the title, dialogue, sound descriptions,
shot prompts and video motion prompts. Preserve supplied proper names and technical IDs.

## Requirements
1. Number shots sequentially from 1. Include every schema field.
2. Describe concrete, filmable actions; preserve story logic and character continuity.
3. Write short English dialogue when appropriate. Use `None` for absent dialogue or
   other inapplicable prose fields. Keep asset URL fields empty as described below.
4. Output only structured data matching the schema, without Markdown fences.
5. Prefer short cinematic shots, usually 2–5 seconds, with natural pacing.
6. Keep `character_1` and `character_2` identifiers consistent across rows. Describe
   each visible character in the corresponding character_description field.
7. Use concise English shot terminology, such as `Close-up / eye level`,
   `Medium shot / low angle`, or `Wide shot / high angle`.
8. Write specific, compact emotion, scene, lighting and sound descriptions.
9. When generating dialogue or narration for video, request spoken English. Do not
   add subtitles or visible text unless the user requests them; requested text is English.

## Image prompt format
Write `shot_prompt` as eight descriptive bracketed sections joined with ` + `:
1. [Composition: framing, camera position, angle and composition]
2. [Character / subject: reuse the character card and describe visible appearance]
3. [Spatial relationships: foreground, midground, background and interactions]
4. [Visible detail: eyes, mouth, posture, clothing, injuries and other visible state]
5. [Environment and props: specific setting and relevant foreground/background objects]
6. [Lighting and atmosphere: direction, color temperature, shadows, fog and rim light]
7. [Visual style: the requested cinematic or illustrative treatment]
8. [Camera settings: lens, aperture, depth of field, motion rendering and grain]

Example:
[Composition: eye-level close-up] + [Character: Alex, an exhausted adult in a dark jacket]
+ [Spatial relationships: seated alone at a desk, monitor to the left]
+ [Visible detail: trembling fingers and unfocused eyes]
+ [Environment and props: late-night office, scattered papers, cold coffee]
+ [Lighting: cool monitor light, soft shadows]
+ [Visual style: realistic cinematic suspense] + [Camera settings: 85mm, f/1.8, shallow depth of field]

## Video motion prompt format
Write `video_motion_prompt` as six bracketed sections joined with ` + `:
1. [Camera movement: direction, speed, strength and stability]
2. [Subject action: specific physical movements or changes]
3. [Environmental motion: wind, clothing, dust, rain, flickering light, etc.]
4. [Sound: ambience, footsteps, breathing, objects and other appropriate sounds]
5. [Dialogue: exact English lines, speaker and delivery; use None when absent]
6. [Duration: 4.0s]

Example:
[Camera movement: very slow, steady push-in]
+ [Subject action: Alex looks up, eyes widening, fingers stopping on the desk]
+ [Environmental motion: paper edges stir in the air conditioning]
+ [Sound: keyboard stops, a distant phone vibrates]
+ [Dialogue: Alex says in English, quietly, "Someone is here."] + [Duration: 4.0s]

## Quality and asset binding
- Avoid generic phrases such as "a person stands", "camera moves", or "complex emotions".
- Do not copy example characters or settings into unrelated stories.
- Always output empty strings for `character_image_1`, `character_image_2`, and
  `reference`: the backend fills these URL slots. Never invent URLs or file names.
- Preserve character identifiers exactly so reference images can be attached.
"""

FREEZONE_VIDEO_STORY_SCRIPT_SYSTEM_PROMPT = FREEZONE_STORY_SCRIPT_SYSTEM_PROMPT + """
## Vision-reference modes
Images may be attached to the request. The task message states which mode applies; follow that
mode and ignore the other one.

### Video-keyframe mode
You are given an ordered set of keyframes sampled from a reference video.
- The story script MUST describe what is actually visible in those frames. Do not invent an
  unrelated story, and do not fall back on the examples in this prompt.
- Read the frames as one continuous clip: identify the real subject(s), setting, wardrobe or
  species, palette, and what physically changes from frame to frame.
- If the subject is not human (animal, object, mascot, animation), say so plainly in
  `character_1` and `character_description_1`. Do not substitute a human character.
- Group consecutive frames into narrative shots rather than describing every frame.
- Set `keyframe_index` on every row to the 1-based index of the input frame that best
  represents that shot. This is how the backend attaches the reference thumbnail.
- Total duration across rows should stay close to the stated video duration.

### Character-reference mode
You are given one portrait-style reference image per character, in the order the task message
lists them. There is no reference video.
- Read every character's real appearance off their image — face, hair, wardrobe, era, species —
  and write `character_description_1` / `character_description_2` from what you actually see.
  Do not describe a character the images do not show.
- Reuse the exact character names given in the task message so the backend can attach each
  character's reference image.
- The story itself comes from the user's request (and the source script, when one is supplied),
  not from the images. The images only fix who the characters are.
- Set `keyframe_index` to 0 on every row: there are no keyframes to attach.
"""

FREEZONE_NODE_TYPE_LABELS: dict[str, str] = {
    "generic": "通用提示词",
    "image": "图片节点提示词",
    "video": "视频节点提示词",
    "audio": "音频节点提示词",
    "text": "文本节点提示词",
}

_translation_agent: Optional[Agent] = None
_text_writer_agent: Optional[Agent] = None
_story_script_agent: Optional[Agent] = None
_video_story_script_agent: Optional[Agent] = None


class FreezoneTranslationResult(BaseModel):
    """Structured translation result produced by the LLM."""

    translated_text: str = Field(description="Translated prompt text.")
    source_language: Literal["zh", "en"] = Field(
        description="Dominant natural language detected from the source text."
    )
    target_language: Literal["zh", "en"] = Field(
        description="Opposite target language used for translation."
    )


def create_freezone_translation_agent() -> Agent:
    """创建 Freezone 中英互译 Agent。"""
    from novelvideo.config import (
        get_newapi_structured_output_model_settings,
        get_newapi_text_pydantic_model,
    )

    model = get_newapi_text_pydantic_model(
        "FREEZONE_TRANSLATION_MODEL",
        FREEZONE_TRANSLATION_MODEL,
        capability="freezone.text.generate",
    )
    return Agent(
        model,
        system_prompt=FREEZONE_TRANSLATION_SYSTEM_PROMPT,
        model_settings=get_newapi_structured_output_model_settings(),
        output_type=FreezoneTranslationResult,
        name="Freezone Prompt Translator",
    )


def get_freezone_translation_agent() -> Agent:
    """获取翻译 Agent 单例。"""
    global _translation_agent
    context = current_model_gateway_context()
    if context is not None and context.is_organization:
        return create_freezone_translation_agent()
    if _translation_agent is None:
        _translation_agent = create_freezone_translation_agent()
    return _translation_agent


def create_freezone_text_writer_agent() -> Agent:
    """创建 Freezone 自由文本生成 Agent。"""
    from novelvideo.config import get_newapi_text_pydantic_model

    model = get_newapi_text_pydantic_model(
        "FREEZONE_TEXT_WRITER_MODEL",
        FREEZONE_TEXT_WRITER_MODEL,
    )
    return Agent(
        model,
        system_prompt=FREEZONE_TEXT_WRITER_SYSTEM_PROMPT,
        output_type=str,
        name="Freezone AI Text Writer",
    )


def get_freezone_text_writer_agent() -> Agent:
    """获取自由文本生成 Agent 单例。"""
    global _text_writer_agent
    if _text_writer_agent is None:
        _text_writer_agent = create_freezone_text_writer_agent()
    return _text_writer_agent


def resolve_freezone_text_writer_model() -> str:
    """返回当前自由文本生成逻辑模型名，供结果与审计记录使用。"""
    from novelvideo.config import get_newapi_text_model_name

    return get_newapi_text_model_name(
        "FREEZONE_TEXT_WRITER_MODEL",
        FREEZONE_TEXT_WRITER_MODEL,
    )


def resolve_freezone_story_script_model(model: str | None) -> dict[str, str]:
    model_text = str(model or "").strip()
    if not model_text:
        return dict(FREEZONE_STORY_SCRIPT_MODEL)
    if model_text == FREEZONE_STORY_SCRIPT_MODEL["id"]:
        return dict(FREEZONE_STORY_SCRIPT_MODEL)
    if model_text.casefold() == FREEZONE_STORY_SCRIPT_MODEL["label"].casefold():
        return dict(FREEZONE_STORY_SCRIPT_MODEL)
    if model_text in LEGACY_FREEZONE_STORY_SCRIPT_MODEL_IDS:
        return dict(FREEZONE_STORY_SCRIPT_MODEL)
    raise ValueError(f"unsupported story script model: {model_text}")


def create_freezone_story_script_agent(model: str | None = None) -> Agent:
    """创建故事脚本生成 Agent。"""
    from novelvideo.api.schemas import FreezoneStoryScriptGenerateData
    from novelvideo.config import (
        get_newapi_structured_output_model_settings,
        get_newapi_text_pydantic_model,
    )

    resolved = resolve_freezone_story_script_model(model)
    llm_model = get_newapi_text_pydantic_model(
        "FREEZONE_STORY_SCRIPT_MODEL",
        resolved["model"],
        capability="freezone.text.generate",
    )
    return Agent(
        llm_model,
        system_prompt=FREEZONE_STORY_SCRIPT_SYSTEM_PROMPT,
        model_settings=get_newapi_structured_output_model_settings(),
        output_type=FreezoneStoryScriptGenerateData,
        # 结构化脚本表字段多、且 shot_no/duration 是严格 int，模型偶尔会把时长写成
        # "2-5"/"3秒" 之类而过不了校验。默认 output_retries=1 只给一次纠正机会不够，
        # 抛 "Exceeded maximum output retries (1)"。对齐本仓其它复杂结构化 agent
        # (episode_planner / content_rewriter)提到 3，让模型按回喂的校验错误自我修正。
        output_retries=model_gateway_output_retries(3),
        name="Freezone Story Script Generator",
    )


def get_freezone_story_script_agent(model: str | None = None) -> Agent:
    """获取故事脚本生成 Agent 单例。"""
    global _story_script_agent
    resolved = resolve_freezone_story_script_model(model)
    context = current_model_gateway_context()
    if context is not None and context.is_organization:
        return create_freezone_story_script_agent(resolved["id"])
    if _story_script_agent is None:
        _story_script_agent = create_freezone_story_script_agent(resolved["id"])
    return _story_script_agent


def build_freezone_translation_task(
    *,
    text: str,
    node_type: Literal["generic", "image", "video", "audio", "text"],
) -> str:
    """构建翻译任务。"""
    node_label = FREEZONE_NODE_TYPE_LABELS[node_type]

    parts = [
        f"Translate the following {node_label}.",
        "You must decide whether the dominant natural language is Simplified Chinese or English.",
        "If dominant language is English, translate into Simplified Chinese.",
        "If dominant language is Simplified Chinese, translate into English.",
        "Do not copy the original prose when translating between different languages.",
        "Preserve IDs, file names, bracket tags, color codes, ratios, and model names exactly, but translate the surrounding natural-language instructions.",
        "Keep it directly usable as a creative prompt.",
    ]
    parts.append(f"Source text:\n{text.strip()}")
    return "\n\n".join(parts)


async def translate_freezone_text(
    *,
    text: str,
    node_type: Literal["generic", "image", "video", "audio", "text"] = "generic",
    egress_context: TrustedEgressContext | None = None,
) -> tuple[str, Literal["zh", "en"], Literal["zh", "en"]]:
    """执行 Freezone 中英互译。"""
    if not text or not text.strip():
        return "", "zh", "en"

    task = build_freezone_translation_task(
        text=text,
        node_type=node_type,
    )
    from novelvideo.model_gateway_runtime import model_gateway_request_scope

    with model_gateway_request_scope(egress_context):
        response = await get_freezone_translation_agent().run(task)
    result = response.output
    target_language: Literal["zh", "en"] = result.target_language
    if target_language == result.source_language:
        target_language = "zh" if result.source_language == "en" else "en"
    return (
        result.translated_text.strip(),
        result.source_language,
        target_language,
    )


async def generate_freezone_text(
    *,
    prompt: str,
    egress_context: TrustedEgressContext | None = None,
) -> tuple[str, str]:
    """根据用户指令生成自由文本，返回逻辑模型名与最终文本。**会出网**。

    形参与 `model_gateway_request_scope` 都照 `translate_freezone_text` 写：
    `runners/freezone.py:FREEZONE_LEAF_EGRESS` 判本函数为 NETWORK 的依据就是它。
    """
    clean_prompt = str(prompt or "").strip()
    if not clean_prompt:
        raise ValueError("prompt is required")

    from novelvideo.model_gateway_runtime import model_gateway_request_scope

    with model_gateway_request_scope(egress_context):
        response = await get_freezone_text_writer_agent().run(clean_prompt)
    generated_text = str(response.output or "").strip()
    if not generated_text:
        raise ValueError("text generation returned empty output")
    return resolve_freezone_text_writer_model(), generated_text


_STORY_SCRIPT_COMMON_RULES = (
    "Include every field: shot number, duration, visual description, both characters and descriptions, "
    "character images, reference, shot type, action, emotion, scene tags, lighting, sound, dialogue, "
    "shot prompt and video motion prompt.",
    "Write every generated prose field in English. Write dialogue and narration in English; "
    "video prompts must request spoken English whenever speech is present.",
    "Follow the user's additional creative requirements.",
    "Output a film-production table, not a prose summary.",
    "Use bracketed sections joined with + for image and video prompts.",
    "Use `None` when dialogue is absent.",
    "Always leave character_image_1, character_image_2 and reference empty; the backend fills "
    "asset URLs. Never invent a URL or filename.",
    "Make image and video prompts detailed enough for generation, not one-line summaries.",
)

_STORY_SCRIPT_STYLE_HINT = (
    "Style guidance:\n"
    "- Sequential shot numbers; most shots last 2–5 seconds.\n"
    "- Use English shot labels such as Close-up / eye level or Medium shot / low angle.\n"
    "- Format character cards as [CharacterID: description]; keep identifiers consistent.\n"
    "- Fill the second character fields when two characters appear.\n"
    "- Image prompts use eight sections: composition, character/subject, spatial relationships, "
    "visible detail, environment/props, lighting, style and camera settings.\n"
    "- Reuse the character card in the subject section; retain camera settings.\n"
    "- Video prompts use six sections: camera movement, subject action, environmental motion, "
    "sound, English dialogue/delivery and duration.\n"
    "- Describe visible physical actions rather than abstract emotional changes."
)


def _character_ref_block(
    character_refs: Sequence[Mapping[str, Any]] | None,
) -> str | None:
    """把角色参考图渲染成任务里的角色卡清单。

    只给模型「名字 + 描述 + 定位」，不给 URL —— 图片本身通过多模态附件传，
    URL 由后端在生成之后按角色名回填，避免模型把链接抄错或凭空编造。
    """
    if not character_refs:
        return None
    lines: list[str] = []
    for index, ref in enumerate(character_refs, start=1):
        name = str(ref.get("name") or "").strip() or f"角色{index}"
        description = str(ref.get("description") or "").strip()
        role = str(ref.get("role") or "").strip()
        detail = "，".join(part for part in (role, description) if part)
        lines.append(f"{index}. {name}" + (f"（{detail}）" if detail else ""))
    return (
        "已提供的角色参考（按顺序对应随附的角色参考图）：\n"
        + "\n".join(lines)
        + "\n请在生成的角色1 / 角色2 字段里使用上面完全一致的角色名，"
        "这样后端才能把对应的角色参考图回填进去。"
    )


def build_freezone_story_script_task(
    *,
    source_text: str,
    prompt: str,
    character_refs: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    """构建故事脚本生成任务。"""
    parts = [
        "根据以下上传剧本内容生成一个完整的故事脚本表。",
        *_STORY_SCRIPT_COMMON_RULES,
    ]
    if prompt.strip():
        parts.append(f"用户要求：\n{prompt.strip()}")
    character_block = _character_ref_block(character_refs)
    if character_block:
        parts.append(character_block)
    parts.append(_STORY_SCRIPT_STYLE_HINT)
    parts.append(f"源剧本内容：\n{source_text.strip()}")
    return "\n\n".join(parts)


def build_freezone_video_story_script_task(
    *,
    frame_count: int,
    prompt: str,
    duration_sec: float | None = None,
    character_refs: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    """构建「视频参考生成分镜脚本」任务。

    与文本模式共用同一张输出表，区别是源素材换成了按时间顺序抽取的关键帧，
    并要求模型给出 ``keyframe_index`` 以便后端回填每一镜的参考图。
    """
    duration_hint = (
        f"该视频总时长约 {duration_sec:.2f} 秒，所有镜头时长加起来应接近这个值。"
        if duration_sec and duration_sec > 0
        else "视频总时长未知，请按关键帧的疏密给出合理时长。"
    )
    parts = [
        f"下面按时间顺序给你 {frame_count} 张从参考视频里抽取的关键帧，"
        "请把这段视频拆解成一张完整的故事脚本表。",
        "这是视频拆解任务，不是原创任务：表格内容必须如实描述这些关键帧里真实出现的"
        "主体、场景、动作和风格。严禁套用系统提示词里的示例角色或示例场景。",
        "先通读全部关键帧判断这究竟是什么内容（人物？动物？动画？实拍？），"
        "再决定角色名和角色描述。主体不是人时，就照实写成该动物 / 物体 / 形象，不要替换成人物。",
        duration_hint,
        "把连续的关键帧归纳成若干叙事镜头，不要逐帧机械罗列。",
        "每一行都必须给出 keyframe_index：最能代表这一镜的输入关键帧序号"
        f"（1 到 {frame_count} 之间的整数）。后端靠它回填这一镜的参考图。",
        *_STORY_SCRIPT_COMMON_RULES,
    ]
    if prompt.strip():
        parts.append(f"用户额外要求：\n{prompt.strip()}")
    character_block = _character_ref_block(character_refs)
    if character_block:
        parts.append(character_block)
    parts.append(_STORY_SCRIPT_STYLE_HINT)
    return "\n\n".join(parts)


def build_freezone_character_story_script_task(
    *,
    image_count: int,
    prompt: str,
    source_text: str = "",
    character_refs: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    """构建「角色参考图生成分镜脚本」任务。

    这一路没有参考视频，只有角色参考图：剧情来自用户提示词（和可选的源剧本），
    图片只负责钉死角色长相，所以 ``keyframe_index`` 一律为 0。
    """
    parts = [
        f"下面按顺序给你 {image_count} 张角色参考图，请据此生成一张完整的故事脚本表。",
        "这是角色参考模式，没有参考视频：角色1 / 角色2 的外貌、服饰、年代、气质"
        "必须照着对应的角色参考图写，不要描述图里没有的人。",
        "剧情本身来自用户要求（以及可选的源剧本），不要凭空照搬系统提示词里的示例剧情。",
        "每一行的 keyframe_index 一律填 0：本模式没有关键帧可以回填。",
        *_STORY_SCRIPT_COMMON_RULES,
    ]
    if prompt.strip():
        parts.append(f"用户要求：\n{prompt.strip()}")
    else:
        parts.append("用户没有额外要求，请围绕这些角色自行编排一段结构完整的短剧。")
    character_block = _character_ref_block(character_refs)
    if character_block:
        parts.append(character_block)
    parts.append(_STORY_SCRIPT_STYLE_HINT)
    if source_text.strip():
        parts.append(f"源剧本内容：\n{source_text.strip()}")
    return "\n\n".join(parts)


async def generate_freezone_story_script(
    *,
    source_text: str,
    prompt: str = "",
    model: str | None = None,
    character_refs: Sequence[Mapping[str, Any]] | None = None,
    egress_context: TrustedEgressContext | None = None,
):
    """执行故事脚本生成（文本 / 角色图模式）。"""
    if not source_text or not source_text.strip():
        raise ValueError("source_text is required")

    task = build_freezone_story_script_task(
        source_text=source_text,
        prompt=prompt,
        character_refs=character_refs,
    )
    from novelvideo.model_gateway_runtime import model_gateway_request_scope

    with model_gateway_request_scope(egress_context):
        response = await get_freezone_story_script_agent(model).run(task)
    return response.output


def create_freezone_video_story_script_agent() -> Agent:
    """创建「视频参考生成分镜脚本」的视觉 Agent。

    走 ``FREEZONE_VISION_MODEL``（``DC-freezone-vision-LLM``）而不是纯文本的
    story-script 别名 —— 带图请求只有视觉渠道能接。
    """
    from novelvideo.api.schemas import FreezoneStoryScriptGenerateData
    from novelvideo.config import (
        get_newapi_structured_output_model_settings,
        get_newapi_text_pydantic_model,
    )
    from novelvideo.official_defaults import DEFAULT_FREEZONE_VISION_MODEL

    return Agent(
        get_newapi_text_pydantic_model(
            "FREEZONE_VISION_MODEL",
            DEFAULT_FREEZONE_VISION_MODEL,
            timeout_seconds_override=300.0,
            capability="vision.analyze",
        ),
        system_prompt=FREEZONE_VIDEO_STORY_SCRIPT_SYSTEM_PROMPT,
        model_settings=get_newapi_structured_output_model_settings(),
        output_type=FreezoneStoryScriptGenerateData,
        output_retries=model_gateway_output_retries(3),
        name="Freezone Video Story Script Generator",
    )


def get_freezone_video_story_script_agent() -> Agent:
    """获取视频分镜脚本 Agent 单例。"""
    global _video_story_script_agent
    context = current_model_gateway_context()
    if context is not None and context.is_organization:
        return create_freezone_video_story_script_agent()
    if _video_story_script_agent is None:
        _video_story_script_agent = create_freezone_video_story_script_agent()
    return _video_story_script_agent


async def generate_freezone_story_script_with_vision(
    *,
    frame_paths: Sequence[str | Path] | None = None,
    character_image_paths: Sequence[str | Path] | None = None,
    source_text: str = "",
    prompt: str = "",
    duration_sec: float | None = None,
    character_refs: Sequence[Mapping[str, Any]] | None = None,
    egress_context: TrustedEgressContext | None = None,
):
    """带图的分镜脚本生成：视频关键帧 / 角色参考图 → 结构化脚本表。

    覆盖两种入口：

    - 「视频参考生成分镜脚本」：``frame_paths`` 是抽出来的关键帧，走视频拆解任务书。
    - 「角色生成分镜脚本」：只有 ``character_image_paths``，走角色参考任务书 ——
      剧情来自 ``prompt``（和可选的 ``source_text``），角色图只负责钉死角色长相。
      这一路不要求 ``source_text``：前端挂了素材时只会发提示词。
    """
    from pydantic_ai import BinaryContent

    from novelvideo.freezone.vision_gateway import load_compact_vision_inputs

    frames = [Path(path) for path in (frame_paths or []) if Path(path).exists()]
    character_images = [
        Path(path) for path in (character_image_paths or []) if Path(path).exists()
    ]
    if not frames and not character_images:
        raise ValueError(
            "vision story script requires at least one keyframe or character image"
        )

    if frames:
        task = build_freezone_video_story_script_task(
            frame_count=len(frames),
            prompt=prompt,
            duration_sec=duration_sec,
            character_refs=character_refs,
        )
    else:
        # 只有角色图：剧情从提示词 / 可选源剧本来，图片只钉角色长相。
        # 这里不能要求 source_text —— 前端在挂了素材时就只发提示词。
        task = build_freezone_character_story_script_task(
            image_count=len(character_images),
            prompt=prompt,
            source_text=source_text,
            character_refs=character_refs,
        )

    vision_inputs = await load_compact_vision_inputs((*frames, *character_images))
    attachments: list[Any] = [
        BinaryContent(data=image.data, media_type=image.media_type)
        for image in vision_inputs
    ]
    from novelvideo.model_gateway_runtime import model_gateway_request_scope

    with model_gateway_request_scope(egress_context):
        response = await get_freezone_video_story_script_agent().run(
            [task, *attachments]
        )
    return response.output


def bind_story_script_assets(
    data: Any,
    *,
    frame_urls: Sequence[str] | None = None,
    character_refs: Sequence[Mapping[str, Any]] | None = None,
) -> Any:
    """把关键帧 / 角色参考图的 URL 回填进生成好的脚本行。

    模型只负责写角色名和 ``keyframe_index``，素材 URL 一律由这里补齐 ——
    这样模型没有机会编造出 404 的链接（issue #207 里角色图列恒为空的另一半原因）。
    """
    frames = [url for url in (frame_urls or []) if url]
    by_name: dict[str, str] = {}
    for ref in character_refs or []:
        name = str(ref.get("name") or "").strip()
        image_url = str(ref.get("image_url") or "").strip()
        if name and image_url:
            by_name[name.casefold()] = image_url
    ordered_images = [
        str(ref.get("image_url") or "").strip()
        for ref in character_refs or []
        if str(ref.get("image_url") or "").strip()
    ]

    def _match(name: str) -> str:
        clean = str(name or "").strip()
        if not clean:
            return ""
        folded = clean.casefold()
        if folded in by_name:
            return by_name[folded]
        # 模型常把角色名写成 `沈昭昭_现代` 这类带状态后缀的稳定 ID，
        # 精确匹配不到时退到包含匹配，仍匹配不到才放弃。
        for candidate, url in by_name.items():
            if candidate in folded or folded in candidate:
                return url
        return ""

    for index, row in enumerate(getattr(data, "rows", []) or []):
        keyframe_index = int(getattr(row, "keyframe_index", 0) or 0)
        if not (1 <= keyframe_index <= len(frames)):
            # 模型没给或给错序号时退回按行号顺序取帧，保证参考列不至于整列为空。
            keyframe_index = index + 1 if index < len(frames) else 0
        row.reference = frames[keyframe_index - 1] if keyframe_index else ""
        row.keyframe_index = keyframe_index

        row.character_image_1 = _match(getattr(row, "character_1", ""))
        row.character_image_2 = _match(getattr(row, "character_2", ""))
        # 只有一张角色图、且模型没写出可匹配的角色名时，直接绑定唯一那张，
        # 否则「角色生成分镜脚本」在模型改写角色名后又会退化成空列。
        if not row.character_image_1 and len(ordered_images) == 1:
            row.character_image_1 = ordered_images[0]

    return data
