# Signify

Speak English, see it signed in ASL by 3D hands.

It works in three steps, each in its own folder:

| Step | Folder | What it does |
| --- | --- | --- |
| 1 | `speech_to_text/` | Listens to the microphone and writes down what was said (vosk, offline) |
| 2 | `text_to_gloss/` | Turns the English sentence into ASL gloss, the ASL word order (spaCy) |
| 3 | `gloss_to_asl/` | Plays each gloss word as a sign on 3D MakeHuman hands |

There are two ways to use it:

- A desktop app, `python main.py`.
- A website: a mic page plus a sign library page. It can be put on Vercel behind a login.

## What you need

- Windows, macOS or Linux, with Python 3.11 or newer
- Git
- Internet for the first setup (downloading models and sign videos)
- A microphone
- About 5 GB of free disk space for the full sign set

## 1. Get the code and make a venv

A venv is a private folder of Python packages just for this project, so nothing clashes with other projects.

Windows (PowerShell):

```bash
git clone https://github.com/prasada29-gif/Sign-app.git signify
```

```bash
cd signify
```

```bash
python -m venv venv
```

```bash
venv\Scripts\activate
```

macOS / Linux:

```bash
python3 -m venv venv && source venv/bin/activate
```

When the venv is on, your prompt starts with `(venv)`. Every command below assumes it is on. Next time you open a terminal, just run the `activate` line again.

If PowerShell says running scripts is disabled, run this once and then activate again:

```bash
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## 2. Install the packages

```bash
python -m pip install --upgrade pip
```

```bash
pip install -r requirements.txt
```

That one file has everything, including spaCy's English model (`en_core_web_sm`). If that model line fails, install the model by hand:

```bash
python -m spacy download en_core_web_sm
```

## 3. Download the speech models (step 1)

```bash
python -m speech_to_text.setup_models
```

This puts the vosk speech model and the silero voice detector in `models/`.

## 4. Download the sign videos and track the hands (step 3)

```bash
python -m gloss_to_asl.setup_clips --max-glosses 0 --max-per-gloss 3 --insecure-host aslsignbank.haskins.yale.edu
```

This downloads WLASL sign videos into `datasets/`. It then tracks the hands in each video and saves the motion in `clips/`.

- `--max-glosses 0` takes all ~2,000 words. This takes hours, and you can stop and re-run it, because finished words are kept. For a quick try, use `--max-glosses 100` instead.
- `--insecure-host` is only there because that one server sends a broken certificate.
- Some videos are gone from the internet. They are skipped and listed in a report.

## 5. Check the signs

```bash
python -m gloss_to_asl.signcheck --apply
```

This compares each tracked sign with ASL-LEX, which it downloads by itself, and marks signs that look wrong. Those words get fingerspelled instead.

## 6. Build the sign pages

```bash
python -m gloss_to_asl.library --web web
```

This writes three pages:

- `exports/library.html`, used by the desktop app
- `web/index.html`, the website mic page
- `web/library.html`, the website sign library

## 7. Run the desktop app

```bash
python main.py
```

Press record, speak, and the 3D hands sign what you said.

## 8. Try the website on your computer

```bash
python -m http.server 8000 --directory web
```

Then open http://localhost:8000 in **Chrome or Edge**.

- The mic button uses the browser's own speech recognition. It only works in Chrome and Edge, and it sends your voice to Google or Microsoft to turn it into text.
- Typing a sentence works in any browser.
- The login gate (`web/middleware.js`) only runs on Vercel, not on this local server.

## 9. Put the website on Vercel, behind a login

The website is public on the internet, but every page asks for one shared username and password first. Only people you give it to can get in.

1. Install the Vercel command line (needs Node.js) and log in:

   ```bash
   npm i -g vercel
   ```

   ```bash
   vercel login
   ```

2. Link the `web` folder to a Vercel project. Run this once and answer the questions:

   ```bash
   vercel link --cwd web
   ```

3. Set the username and password. Vercel asks you to type each value, and they never go into the code or git:

   ```bash
   vercel env add SITE_USER production --cwd web
   ```

   ```bash
   vercel env add SITE_PASSWORD production --cwd web
   ```

4. Build the pages (step 6), then deploy:

   ```bash
   vercel deploy --prod --cwd web
   ```

5. Open the link Vercel prints. The browser asks for the username and password.

To change the password, remove it with `vercel env rm SITE_PASSWORD production --cwd web`, add the new one, and deploy again.

If `SITE_USER` or `SITE_PASSWORD` is missing, the site lets nobody in. So a deploy can never be open by accident.

## Tests

The `tests/` folder is kept on the main dev computer, not in git. Where it exists, run:

```bash
python -m pytest
```

## Optional tools

These are in `gloss_to_asl/tools/`. You only need them to rebuild things that are already in the repo:

- `fill_hands.py` and `reextract.py`: re-track hands for signs that came out badly
- `build_nouns.py`: rebuild the noun list (`pip install nltk` first)
- `build_hand.py`: rebuild the 3D hand from the MakeHuman source files (kept on one computer, not in git)

## Licenses and data: read before sharing

- **3D hands:** MakeHuman base mesh, skeleton and weights, CC0, free to use.
- **Sign motion:** tracked from WLASL videos, which allow research and academic use only.
  - The WLASL videos and the motion made from them are not in git (`datasets/`, `clips/`, `exports/`, `web/*.html` are ignored), and they must not be shared openly.
  - That is why the website sits behind a login.
- **Sign checking:** ASL-LEX, used for research.
- Translations are made by a program and have not been checked by a Deaf signer.
