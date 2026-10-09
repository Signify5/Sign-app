# Sign Playback Step Specification (Step 3)

## Goal

Given an ordered list of ASL glosses, find a matching video clip for each and play them back in order. Step 2 (owned by a teammate) produces the gloss list; step 3 consumes it. Step 3 must not depend on step 2's implementation, only on the input contract below.

## Input Contract (proposed, to be agreed with the step 2 owner)

- `play(glosses: list[str])` / `build_playlist(glosses: list[str])`.
- Each gloss is an uppercase string, e.g. `["HELLO", "MY", "NAME", "VEDANT"]`.
- Step 3 owns all lookup, normalization (case, trimming), fallback, and error reporting. Step 2 does not need to know which words have clips.
- If step 2 later needs richer fields (timing, non-manual markers), the contract is extended then; out of scope now.

## Data Sources

- **Primary: WLASL.** Per-gloss clips with many signings (different signers) per word.
- **Letters only: ASL Signbank.** Used solely for A-Z (and 0-9 if available) fingerspelling clips. This is the only Signbank use.
- ASLLRP is out of scope.
- ASL Citizen is not used: its licence forbids sharing the data or anything made from it (including motion tracked from its videos), and the library must ship as an app.
- Dataset licences are academic/non-commercial. Clips and the index are gitignored and never redistributed in the repo.

### Acquisition

- `setup_clips.py` (mirrors `setup_models.py`): reads the WLASL metadata JSON, downloads every clip whose URL is still alive for the chosen vocabulary, records dead/failed links in a report, and writes a local index.
- Clips live in a gitignored folder, e.g. `clips/`. The index maps `gloss -> [clip entries]`, each with signer id (where known), resolution, fps, duration, and path.
- Must be re-runnable (skips files already downloaded and verified).
- Fingerspelling letters are fetched the same way from ASL Signbank.

## Core Module (Qt-free)

Mirrors step 1's `SignifySession`: the logic is a plain Python module, the GUI only consumes it.

- `ClipIndex`: loads the index, answers `lookup(gloss) -> list[Clip]`.
- `build_playlist(glosses) -> Playlist`: ordered list of items. Each item is either a `Clip` (a WLASL sign) or a fingerspelled run (a letter-clip sequence), plus the original gloss for reference.
- Status events via the same callback style as step 1 (`on_status`): missing gloss, fingerspell fallback used, empty playlist, index not found.

### Selection rule (many signings per gloss)

Deterministic: pick the best-quality clip by a fixed ranking (highest resolution, then longest-covered signer, then lowest clip id as the tiebreaker). Same input always yields the same playlist, which keeps tests and demos reproducible. A "prefer one signer per playlist" preference is a possible later refinement, not part of this step.

### Missing glosses

If a gloss has no WLASL clip, fingerspell it letter by letter using the ASL Signbank letter clips. Any character with no letter clip is skipped and reported through `on_status`. The playlist is still produced, never an exception for a missing word.

## Playback and Export (two consumers of the same Playlist)

Signs are shown as a synthetic digital pair of hands, not as video of a person. Each playlist clip is a motion file (hand landmarks); `motion.compose` chains them into one `Timeline`, and `render.render_frame` draws each frame.

1. **Qt player window (PySide6, QLabel + QTimer)**: plays the timeline with pause/resume/stop, a current-gloss label, an inline status banner (same style as step 1, no modal dialogs), a speed control, a gap setting, a mirror toggle and a faint head-and-shoulders guide toggle.
2. **MP4 export**: renders the same timeline to an H.264 MP4 through the ffmpeg binary bundled with `imageio-ffmpeg` (no system ffmpeg needed). Speed, gap and mirror apply to the export too.

## Acceptance Criteria

1. `setup_clips.py` builds a local WLASL clip set and index and reports which links were dead.
2. `build_playlist(["HELLO", "WORLD"])` returns clips in the given order, deterministically.
3. A gloss missing from WLASL falls back to fingerspelling; status events say so.
4. A gloss that cannot be played at all is reported, and the rest of the playlist still plays.
5. The Qt player plays a playlist end to end, in order, with working pause/stop, speed control, and gap setting.
6. The exporter writes one MP4 of the whole playlist at a fixed resolution and fps.
7. No network access is needed after `setup_clips.py` has run.
8. Step 3 can be driven from a plain gloss list with no dependency on step 2's code.
9. A pytest suite (convention from step 1) covers selection determinism, fingerspell fallback, missing-gloss status events, and playlist ordering, using a fake index/clips so no dataset or ffmpeg is required.

## Out Of Scope

- Producing gloss from English (step 2).
- ASLLRP data.
- Grammar/non-manual-marker rendering, or facial expression; hands only (planned for phase 3, below).
- Sign recognition (video to text).
- Redistributing the dataset.
- Mac/Linux verification (code stays cross-platform; verified on Windows only, as in step 1).

## As Built (deviations and findings)

- Pipeline: `setup_clips.py` downloads a WLASL video, `animate.extract_motion` tracks both hands (MediaPipe Hand Landmarker) and shoulders (Pose Landmarker lite), normalizes landmarks to body units (origin = mid-shoulder, 1.0 = shoulder width), resamples to 30 fps and saves `clips/motion/<gloss>/<id>.npz`. The index (`clips/index.json`) points at those files. Models auto-download to `models/`.
- Modules (all in `gloss_to_asl/`, run from the repo root with `python -m gloss_to_asl.<name>`): `clips.py` (Clip, ClipIndex), `playlist.py` (build_playlist), `motion.py` (Motion, Timeline, compose), `animate.py` (tracking), `setup_clips.py`. The first 2D player (`render.py`, `export.py` for MP4, `player.py`) was replaced by the 3D hands and removed.
- Selection: clips rank by sign-check faults (fewest first, see below), then tracking quality (share of frames with a hand), then signer coverage, then clip id.
- Transitions: hands glide (eased, 0.3 s) from the previous sign's end pose or from a rest pose below the frame into the next sign's first pose; a hand a sign does not use rests. The hands are drawn 1.5x life size so finger shapes stay readable.
- Letter clips: the ASL Signbank letter video ids could not be discovered programmatically. Fingerspelling resolves a letter from (1) `clips/letters/X.mp4` (dropped in, or downloaded from an optional `letters_manifest.json` of `{letter: url}`; processed into motion at setup), else (2) a one-letter WLASL gloss. WLASL has only 21 of 26 letters (missing C, L, X, Y, Z).
- Download reality: of the first 10 WLASL glosses, ~130 of ~150 instances were skipped up front (YouTube, .swf) and about half of the rest failed (404, 403, undecodable files, no hands found). Setup falls through to the next signing per gloss and logs failures to `clips/setup_report.json`.
- aslsignbank.haskins.yale.edu (about 1070 WLASL instances) serves an incomplete TLS chain. `--insecure-host HOST` skips verification for that host only; it is opt-in.
- Known limits: depth is not tracked (2D landmarks only), so hand contact and some orientations look approximate; no facial grammar. Sign correctness rests on WLASL gloss labels and has not been checked by an ASL signer.
- Later option: a 3D rigged hand fed by the same motion files.
- 3D hands (`hand3d.py`): the MakeHuman hand (CC0) is driven by MediaPipe world landmarks, retargeted per bone. The page shows hands only, with a fixed camera fitted to the whole sequence.
- Fingerspelling (`fingerspell.py`): A-Z are authored handshapes (per-finger flex and spread, a thumb target solved by a small IK, palm orientation); J and Z trace their paths. These replace the unreliable WLASL letter clips in the 3D player (`build_playlist(..., authored_letters=True)`). The right hand spells, each letter is held 0.28 s, and a doubled letter slides sideways.
- Library page (`library.py`): every gloss's best clip (1974 signs, one-letter glosses left to the alphabet except the pronoun I) plus the alphabet, in one page of about 4 MB. Signs are stored at 15 fps, quantised and delta-coded. The page chains them in the browser, so a typed sentence plays: multiword signs (THANK YOU, HARD OF HEARING) match first, plurals fall back to the singular, numbers are signed digit by digit past ten, English-only words (the, is, ...) are dropped, and anything else is fingerspelled.
- Word aliases (`aliases.py`): about 5700 more English words play existing signs. Hand-written tables cover same-sign synonyms (HI = HELLO), irregular verbs and plurals (WENT = GO), contractions (CAN'T = CANNOT, WON'T = REFUSE) and compounds (DRIVER = DRIVE + AGENT); every sign and alias also gets its regular -s, -ing, -ed and -ly forms. A word with its own WLASL sign always keeps it. The page lists the substitutions under the sentence ("Signed as: went = GO").
- ASL word order (`assets/hand/gloss.js`, inlined into the page, tested with node): a typed English sentence is put into ASL order before it plays. English-only words are dropped (articles, forms of be, helper DO, TO, AT, OF); GOING TO + verb is WILL and HAVE TO is MUST; time signs and phrases go first (YESTERDAY I GO STORE, NEXT WEEK, TWO DAY AGO) and make WILL redundant; a past verb with no time sign ends the clause with FINISH, a past state starts with PAST; WH-questions end with the question sign (YOUR NAME WHAT); a plural noun is signed twice with a 4 cm sideways shift unless a number or quantity sign comes before it (THREE CAT). Nouns are told from verbs by `assets/hand/nouns.txt` (WordNet sense counts, `tools/build_nouns.py`). The page shows the resulting order.
- Hand labels (`animate.assign_hands`): the hand tracker's own left/right label flips between frames and can give both hands one label, which dropped a hand or swapped the hands mid-sign. Each detected hand is now matched to the nearer arm's wrist from the pose tracker, else to where that hand just was. Re-tracking all clips (`tools/reextract.py`) cut signs with a hand jumping across the body from 113 to 3.
- Sign checks (`signcheck.py`): every signing is checked against ASL-LEX 2.0 (Caselli et al. 2017, Sehyr et al. 2021, CC BY 4.0; `datasets/asllex/`), which codes each sign's type (one hand, two hands), place (head, body, hand, neutral space) and selected fingers. Serious faults ("bad", 10 points each) are tracking failures that play visibly wrong: too few frames, no hand raised, a hand jumping across the body, or a hand the tracker lost while the pose tracker shows that arm raised (arm positions cached in `clips/pose/`). Mismatches with ASL-LEX are warnings (1 point): ASL-LEX holds one way of signing each word and WLASL signers often use another (SOON, GRAB, EXPERT), so they choose between signings but never drop a word. Without an ASL-LEX entry, Battison's symmetry condition is checked (both hands moving share a handshape). With three or more signings, one that moves unlike the others is flagged. `signcheck.py --apply` stores the points in the index; up to three signings per gloss are fetched (`setup_clips.py --max-per-gloss 3`) so there is a choice. `library.py` leaves out a gloss whose best signing still has a serious fault (`--keep-failed` keeps it), and the word is spelled.
- Lost hands (`animate.fill_lost_hands`): MediaPipe drops a hand when the hands touch or cross (SCHOOL, STOP, MARRY, CONGRATULATIONS). Where the pose tracker still shows that arm raised, the gap is filled: the handshape blends between the frames either side and the wrist follows the pose wrist plus the offset seen at those frames. `tools/fill_hands.py` applied this to the stored motion (46054 frames in 4673 of 5458 signings).
- Results of the checks: 5458 signings of 1994 glosses. Serious faults fell from 175 (one signing per gloss) to 31 with alternate signings and to 2 after the fill; PENNY and PERSONALITY are left out and spelled. About 160 ASL-LEX warnings remain, mostly regional or older variants. No fluent signer has reviewed the signs yet.
- Agent ending: person nouns (TEACHER, DRIVER, AMERICAN) are the verb or place plus the agent ending, both flat hands moving down the sides. WLASL has no clip of the ending alone, so `library.agent_track` builds AGENT from the PERSON sign's movement with B hands.
- Word forms: a noun sign gets only its plural unless ASL signs the action the same way (`aliases.VERB_TOO`: STUDIED, RAINING); otherwise FIRED would play flames and BOOKED a book. Words whose WLASL sense is unclear lose their derived forms (MEANT, MEANING). English's perfect (I have eaten) drops HAVE and is marked past like any past verb.
- Step 2 joined (`translate.py`, `sign_panel.py`): step 2's spaCy reading (from `gloss.py`) and step 3's sign rules (gloss.js) are combined, not either one alone. `translate.analyze` gives each word gloss.js reads its dictionary form, part of speech, tense tag and negation; `toGloss(text, lex, nlp)` then uses spaCy to tell noun from verb (my BOOKS is a plural, he BOOKS a room is not), to find past tense and plurals, and to fall back to the dictionary form when a word has no sign of its own (BIGGER plays BIG), but never to a noun-only sign for a verb (FIRED is spelled, not FIRE). If the readings do not line up word for word, or spaCy is not installed, gloss.js uses its own rules (the shareable page always does). `gloss.to_gloss` still works on its own and its 12 examples are tests; its `FS:WORD` items are spelled by `build_playlist` unless WORD has a sign.
- Main window (`gui.py`): the speech-to-text window has the 3D hands beside the transcript (Qt WebEngine showing `exports/library.html`); each finished sentence from vosk is signed. The page loads three.js from a CDN, so the hands need internet. Build the page first with `python -m gloss_to_asl.library`.
- Not possible with hands only and text input: facial grammar (raised or lowered brows for questions, negation headshake, mouth morphemes), spatial referencing (placing people in space, directional verbs), classifiers and verb aspect. Yes/no questions therefore look like statements. These need a face/body and real scene understanding (step 2's richer contract).

## Phase 3: Face (planned)

ASL grammar is carried partly by the face and head, which the hands-only player cannot show. Phase 3 adds a face.

- **What it must show** (non-manual markers):
  - Yes/no questions: raised eyebrows, head tilted forward, held over the whole question.
  - WH-questions: lowered, drawn-together eyebrows over the whole question.
  - Negation: head shake over the negated part (NOT, NEVER, DON'T WANT).
  - Topic: raised eyebrows over a topic moved to the front of the sentence.
  - Conditionals (IF ... THEN): raised brows on the condition.
  - Mouth morphemes that change meaning: MM (normally, regularly), CHA (big), OO (small), CS (recently, near), TH (carelessly), PAH (finally).
  - Signs whose face is part of the sign (lexical expressions), such as NOT-YET (tongue), and the affect a sign carries (HAPPY, SAD, ANGRY).
- **Model:** the MakeHuman head and upper body (CC0), so it matches the hands. Facial movement through blendshapes (brows, eyelids, mouth, cheeks, tongue) and a neck/head joint for nods, shakes and tilts.
- **Where the motion comes from:**
  - Per sign: MediaPipe Face Landmarker (52 ARKit-style blendshapes plus head rotation) on the same WLASL videos, run alongside the hand tracking, so each sign keeps its own lexical face.
  - Per sentence: the grammar markers above are layered over the signs by rule, from `gloss.js`'s output (it already knows which sentences are questions and which part is negated). A marker spans its whole phrase, eases in before the first sign and out after the last.
- **Contract change:** step 2's gloss list grows optional non-manual spans, e.g. `{"gloss": "NOT", "nmm": ["headshake"]}`, or sentence-level markers (`"type": "wh-question"`). Plain gloss lists keep working.
- **Body:** shoulders and torso turns (role shift, placing people left and right) come with the upper body. Spatial referencing and directional verbs need the gloss list to name locations and are a later step.
- **Licences:** the same as now. WLASL-derived face motion is non-commercial and stays out of git and the published page builds that exclude it. MakeHuman assets are CC0.
- **Acceptance:** a question plays with raised (yes/no) or lowered (WH) brows over the whole question; NOT plays with a head shake; signs with a lexical face show it; no expression shows on a sentence that should have none; an ASL signer reviews a set of sentences.

## Open Items

- Agree the gloss input contract with the step 2 owner (default above). Done in part: see "Step 2 joined".
- Decide the initial vocabulary subset for download (full WLASL vs a smaller subset to keep setup fast).
- Confirm WLASL and Signbank licence terms fit the intended use.
- Have an ASL signer review the rendered signs.
