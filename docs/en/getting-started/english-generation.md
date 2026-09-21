# English generation

This installation defaults to English for both the interface and new generated
creative content. The backend uses `GENERATION_LANGUAGE=en` by default. The local
`.env` also explicitly selects it.

New script-generator tables request English titles, shot descriptions, dialogue,
image prompts, video motion prompts and sound descriptions. The same language
policy is applied to normal and streaming text-model requests. Technical field
names, enum values, IDs, reference URLs and supplied proper names stay intact.
The explicit Chinese/English translation tool remains bidirectional.

Character, scene and episode generation use the configured output language rather
than assuming Chinese from short or ambiguous source text. Canvas video requests
ask for English when speech or on-screen text is requested; they do not add speech
to otherwise silent clips. Music requests with vocals also request English.

## Audio

Use English dialogue or narration as the speech input. The voice-cloning workflow
reads the supplied text; it does not translate existing recordings or authored
Chinese dialogue. Choose an English reference voice in the audio/character voice
settings for the intended accent. Newly initialized canvas audio nodes prefer an
available English-labeled reference. Explicitly selected voices remain selected.
The Edge TTS fallback defaults to `en-US-JennyNeural`.

## Existing projects

Previously saved Chinese scripts, prompts, clips and recordings are unchanged.
Regenerate the script from your English source, review the new English dialogue,
and regenerate the affected video and audio. Regeneration uses the selected
model provider and its credits. Already-running jobs may retain their previous
request.

## Configuration

```dotenv
GENERATION_LANGUAGE=en
EDGE_TTS_VOICE=en-US-JennyNeural
```

Restart the backend after changing these values. `GENERATION_LANGUAGE=auto`
restores source-language selection for episode/asset generation; `zh` requests
Chinese. The canvas script-generator templates and UI in this installation are
English-first. Model outputs depend on provider compliance: review dialogue,
subtitles, pronunciation and lip sync before exporting a final video.
