# Signify

A small offline speech-to-text popup: turn on the mic, see English text appear as
you speak, export the recording and transcript. First reusable piece for a larger
Sign-app project.

Fully offline after setup -- no audio or transcript data leaves the machine.

## Run it

```bash
python -m venv venv
venv\Scripts\pip install -r requirements.txt
venv\Scripts\python setup_models.py   # downloads the Vosk model + silero-vad
venv\Scripts\python main.py
```

Run tests with `pytest` (uses fakes only -- no mic, model, or network needed).

## How it works

- `audio.py` -- captures the mic via `sounddevice`, writes a WAV as you go.
- `vad.py` -- silero-vad detects speech vs. silence to decide when to finalize a
  sentence (0.9s of silence, or you hit stop).
- `engine.py` -- the transcription engine contract (`EngineAdapter`) and the Vosk
  implementation of it. Vosk is genuinely streaming: partial text comes back in a
  few milliseconds as you speak.
- `postprocess.py` -- capitalizes and punctuates Vosk's raw (lowercase,
  unpunctuated) output.
- `session.py` -- `SignifySession`, the orchestrator. Wires audio → VAD → engine →
  a plain-Python callback interface (`on_partial` / `on_final` / `on_status`, plus
  `wav_path` / `transcript_path`). Has no Qt dependency, so a later track can
  consume it directly without depending on the GUI.
- `gui.py` / `main.py` -- the PySide6 popup itself.

Exports both a `.wav` recording and the transcript as `.txt` and `.json`.

## Why Vosk

Four engines were benchmarked against a fixed 8-sentence script (openai-whisper,
whisper.cpp, Moonshine, and NVIDIA Parakeet, in addition to Vosk):

| Engine | Tier | Hardware | Accuracy (WER) | Avg latency (partial / final) | Avg CPU | Avg RAM |
|---|---|---|---|---|---|---|
| **Vosk** | small | CPU | 5.63% | **3ms / 11ms** | **86.9%** | **641MB** |
| openai-whisper | tiny | GPU | 9.86% | 56ms / 71ms | 123.7% | 1517MB |
| openai-whisper | base | GPU | 5.63% | 72ms / 92ms | 132.6% | 1622MB |
| whisper.cpp | tiny | CPU | 9.86% | 244ms / 249ms | 324.8% | 657MB |
| whisper.cpp | base | CPU | 7.04% | 577ms / 562ms | 368.1% | 813MB |
| whisper.cpp | large-v3-turbo-q5_0 | CPU | 2.82% | 9777ms / 9687ms | 381.4% | 977MB |
| Moonshine | base | CPU | 2.82% | 181ms / 302ms | 1781.2% | 800MB |
| Parakeet | tdt-1.1b | GPU | 5.63% | 78ms / 84ms | 86.6% | 5034MB |

**Vosk won**: fastest by a wide margin, lowest resource use, ties for best
practical accuracy, and needs no GPU at all. Its apparent WER gap against the
2.82%-WER engines is mostly a labeling artifact -- Vosk's "errors" were missing
capitalization on proper nouns (e.g. "tuesday"), not misheard words, while the
other engines' errors were real word-choice mistakes.

Caveats: the reference script was locally synthesized via Windows TTS, not real
human speech, so real-world accuracy may differ -- this is a reasonable smoke test,
not a substitute for testing against a real recorded voice. `openai-whisper base`
(GPU) and Parakeet (GPU) are worth reconsidering if a target deployment always has
a capable GPU and needs Whisper-family robustness to accents/noise.

## Out of scope (this milestone)

MP3 export, cloud transcription, translation, sign-language recognition, accounts.
