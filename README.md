# Signify

A small offline speech-to-text popup. Hit start, watch English text show up as you
talk, then export the recording and transcript when you're done. It's the first
piece of a bigger sign-language app I'm building.

Everything runs locally after the initial setup -- no audio or transcript ever
leaves your machine.

## Running it

```bash
python -m venv venv
venv\Scripts\pip install -r requirements.txt
venv\Scripts\python setup_models.py   # grabs the Vosk model + silero-vad
venv\Scripts\python main.py
```

Tests run with `pytest` and don't need a mic, model, or network connection -- they're
all fakes.

## What's in here

- `audio.py` -- grabs mic audio through `sounddevice` and writes it to a WAV as it goes.
- `vad.py` -- uses silero-vad to tell speech apart from silence, so it knows when to
  wrap up a sentence (after 0.9s of quiet, or when you hit stop).
- `engine.py` -- defines the transcription engine interface and the Vosk backend that
  implements it. Vosk streams properly, so partial text shows up within milliseconds.
- `postprocess.py` -- adds capitalization and punctuation to Vosk's raw output, which
  otherwise comes back lowercase with no punctuation at all.
- `session.py` -- `SignifySession`, the piece that wires audio, VAD, and the engine
  together and exposes plain callbacks (`on_partial` / `on_final` / `on_status`, plus
  `wav_path` / `transcript_path`). No Qt in here, so it can be reused outside the GUI.
- `gui.py` / `main.py` -- the actual PySide6 popup window.

Each session exports a `.wav` recording plus the transcript as both `.txt` and `.json`.

## Why Vosk

I benchmarked four other engines against Vosk on a fixed 8-sentence script:
openai-whisper, whisper.cpp, Moonshine, and NVIDIA Parakeet.

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

Vosk won by a wide margin: fastest, lightest on resources, ties for best real-world
accuracy, and doesn't need a GPU at all. Its WER looks a bit worse than the top two
engines on paper, but that gap is mostly a labeling quirk -- Vosk's "mistakes" were
missing capitalization on proper nouns (like "tuesday"), not actual misheard words,
while the other engines made real word-choice errors.

One caveat: the test script was synthesized with Windows TTS, not real human speech,
so this is a decent smoke test but not a substitute for trying it on an actual voice.
If a deployment always has a solid GPU and needs extra robustness to accents or
background noise, `openai-whisper base` or Parakeet are worth a second look.

## Not doing (yet)

MP3 export, cloud transcription, translation, sign-language recognition, accounts.
