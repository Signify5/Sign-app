# Signify

Speak English, see it signed in ASL by 3D hands.

It works in three steps, each in its own folder:

| Step | Folder | What it does |
| --- | --- | --- |
| 1 | `speech_to_text/` | Listens to the microphone and writes down what was said (vosk, offline) |
| 2 | `text_to_gloss/` | Turns the English sentence into ASL gloss, the ASL word order (spaCy) |
| 3 | `gloss_to_asl/` | Plays each gloss word as a sign on 3D MakeHuman hands |

The hand motion for all 1,973 signs is already in the repo, in `gloss_to_asl/assets/signs.json.gz`.

## Setup and run

```bash
git clone https://github.com/Signify5/Sign-app.git signify
```

```bash
cd signify
```

Windows:
```bash
python setup.py
```

Mac or Linux:
```bash
python3 setup.py
```

Windows:

```bash
venv\Scripts\python main.py
```

macOS / Linux:

```bash
venv/bin/python main.py
```

Press record, speak, and the 3D hands sign what you said.

## The website

Setup also builds a website in `web/index.html`: speak or type a sentence and the hands sign it. To try it on your computer:

```bash
venv\Scripts\python -m http.server 8000 --directory web
```

Then open http://localhost:8000 in **Chrome or Edge**.

- The mic button uses the browser's own speech recognition. It only works in Chrome and Edge, and it sends your voice to Google or Microsoft to turn it into text.
- Typing a sentence works in any browser.

### Grammar service

The page can ask a small spaCy service to analyse each sentence, which gives better word order and tenses. If the service is slow or down, the page falls back to its built-in rules. The service lives in `gloss_service/` and runs on Vercel:

```bash
cd gloss_service
vercel deploy --prod
```

Set `GLOSS_API` to the service address when building the page (the Pages workflow does this). Leave it unset to use the built-in rules only.

### Host on GitHub Pages

This repo includes a GitHub Actions workflow at `.github/workflows/deploy-pages.yml` that builds and deploys `web/` to GitHub Pages on every push to `main`.

To enable it:

1. Open **Settings → Pages** in the repository.
2. Set **Source** to **GitHub Actions**.
3. Push to `main` (or run the workflow manually from the Actions tab).

After deployment, the site is public on GitHub Pages with no password gate.

## Licenses and data: read before sharing

The code is MIT licensed (see `LICENSE`). The sign data has separate terms. Every credit and license is listed in `NOTICE.md`.

- **3D hands:** MakeHuman base mesh, skeleton and weights, CC0, free to use.
- **Sign motion:** tracked by this project from WLASL sign videos.
  - WLASL is shared under the Computational Use of Data Agreement (C-UDA 1.0), which puts no limits on Results, meaning things made from the data that don't contain it.
  - `signs.json.gz` holds only joint angles and positions, with no video frames, which is why it is in the repo.
  - WLASL asks for academic and non-commercial use only. Do not sell Signify or use it commercially.
  - The WLASL videos themselves (`datasets/`) and the full tracking output (`clips/`) stay out of git.
  - Credit: Li et al., "Word-level Deep Sign Language Recognition from Video", WACV 2020 (https://github.com/dxli94/WLASL).
- **Sign checking:** ASL-LEX, used for research.
- Translations are made by a program and have not been checked by a Deaf signer.
