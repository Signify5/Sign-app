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
