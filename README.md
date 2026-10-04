# Tech Viz Bot — Autonomous Daily Tech-Video Generator

A fully automated pipeline that picks a technology topic, writes a script with an
LLM, renders **code-driven visualizations** (Pillow + NumPy + Matplotlib in a
3Blue1Brown / Kurzgesagt style), generates a Vietnamese voice-over, burns in
subtitles, composites everything with FFmpeg, and uploads the finished video to
YouTube — twice a day, unattended, on free GitHub Actions runners.

No VPS, no manual steps, no LaTeX, no headless browser. Everything runs in pure
Python plus FFmpeg.

---

## Table of Contents

- [Highlights](#highlights)
- [How It Works](#how-it-works)
- [Tech Stack](#tech-stack)
- [The `mathviz` Animation Engine](#the-mathviz-animation-engine)
- [Multi-Tier Fallbacks](#multi-tier-fallbacks)
- [Local Setup](#local-setup)
- [API Keys](#api-keys)
- [YouTube OAuth (one-time)](#youtube-oauth-one-time)
- [Running Locally](#running-locally)
- [Deploying to GitHub Actions](#deploying-to-github-actions)
- [Configuration Reference](#configuration-reference)
- [Project Layout](#project-layout)
- [Security Model](#security-model)
- [Troubleshooting](#troubleshooting)

---

## Highlights

- **End-to-end automation** — topic selection → script → animation → TTS →
  subtitles → compositing → thumbnail → upload, with zero human input.
- **Three video formats** — long 16:9 explainer, vertical 9:16 Short (up to 3 mins / 180s), and
  **mega 16:9 deep-dive** (15–30+ minutes) with chapter-based hierarchical scripting.
- **High animation density ($\ge 70\%$) & instant visual hook** — animations start
  right from second 0 (Scene 1), prioritizing AI-generated Python/Manim animations.
- **AI-coded Light Neon Manim animations** — generates stunning cyberpunk/dark-mode
  visuals (glowing vectors, HUD, neon lines) via Manim Community Edition without
  any LaTeX dependency, with automatic AI self-repair on syntax/render errors.
- **Code-driven `mathviz` engine** — a lightweight Pillow + NumPy engine rendering
  smooth mathematical and system diagrams with stable framing (zoom locked to 1.0
  to prevent content clipping).
- **Multi-tier image search** — Google Custom Search API → DuckDuckGo / Bing 16:9
  web search (100% free, no key needed) → Pexels → Openverse.
- **Resilient by design** — 10-tier LLM fallback, multi-tier TTS, and multi-tier
  image/video fallbacks keep the pipeline running indefinitely.
- **Cost-aware** — runs 100% free on GitHub Actions using free-tier LLMs, Edge-TTS,
  and keyless image fallbacks.

---

## How It Works

```
┌─────────────┐   ┌──────────────┐   ┌───────────────────┐   ┌───────────┐
│ Topic       │──▶│ Script       │──▶│ Scene rendering    │──▶│ Voice-over│
│ selection   │   │ (LLM)        │   │ (mathviz/Pillow/   │   │ (TTS)     │
│ (LLM + DB   │   │ narration +  │   │  Matplotlib) +     │   │           │
│ dedup)      │   │ visuals +    │   │  b-roll footage    │   │           │
│             │   │ exercises    │   │                    │   │           │
└─────────────┘   └──────────────┘   └───────────────────┘   └─────┬─────┘
                                                                    │
        ┌───────────────────────────────────────────────────────────┘
        ▼
┌─────────────┐   ┌──────────────┐   ┌───────────────────┐   ┌───────────┐
│ Subtitles   │──▶│ Compositing  │──▶│ Metadata +        │──▶│ YouTube   │
│ (Whisper)   │   │ (MoviePy +   │   │ AI thumbnail      │   │ upload    │
│             │   │  FFmpeg)     │   │ (Kurzgesagt style)│   │ (Data API)│
└─────────────┘   └──────────────┘   └───────────────────┘   └───────────┘
```

Pipeline stages (all orchestrated by [`src/pipeline.py`](src/pipeline.py)):

1. **Topic selection** — `topic_selector.py` asks the LLM for a fresh topic within
   configured domains, checking `output/state.db` to avoid repeats within
   `dedup_days`.
2. **Script writing** — `script_writer.py` prompts the LLM for a structured
   `Script` (title, scenes with narration + visual spec, optional practice
   exercises). Animation scenes are emitted as **declarative JSON specs**.
3. **Scene rendering** — for each scene:
   - A static PNG is always rendered as a fallback (`visual_engine.render_scene`).
   - TTS is synthesized first so the exact audio duration drives animation length.
   - Animation scenes are built via `animation_bridge` + the `mathviz` engine.
   - Optional real b-roll footage (Pexels Videos) is fetched for dynamic backdrops.
4. **Voice-over** — `tts.py` synthesizes narration, locking one voice per video.
5. **Subtitles** — `subtitles.py` uses faster-whisper to time captions and burns
   them in.
6. **Compositing** — `compositor.py` sequences scenes, crossfades, background
   music, transition SFX, and optional intro/outro with MoviePy + FFmpeg.
7. **Thumbnail + metadata** — `thumbnail_ai.py` generates a Kurzgesagt-style AI
   background and overlays the title; `metadata.py` builds the YouTube title,
   description, and tags.
8. **Upload** — `youtube_uploader.py` uploads via the YouTube Data API v3 and
   records the result in the DB.

---

## Tech Stack

| Component      | Technology                                                           |
| -------------- | -------------------------------------------------------------------- |
| LLM            | 10-tier fallback: Anthropic (Claude Opus) → Gemini → Groq → OpenRouter → Z.ai → Mistral → NVIDIA NIM → GitHub Models → SambaNova → Cloudflare |
| TTS            | VieNeu-TTS → Edge-TTS (free); ElevenLabs available but disabled by default |
| AI Animation   | Manim Community Edition (Light Neon / Cyberpunk style, zero LaTeX) with AI self-repair |
| Math Animation | Self-written `mathviz` (Pillow + NumPy, zoom locked to 1.0)           |
| Static visuals | Matplotlib (mathtext formulas, pure Python — no LaTeX)              |
| Images         | Google Custom Search → DuckDuckGo / Bing 16:9 Web Scrape → Pexels → Openverse |
| Video b-roll   | Pexels Videos (dynamic background clips)                             |
| Thumbnail      | HuggingFace FLUX.1-schnell → Pollinations Flux (keyless fallback)   |
| Subtitles      | faster-whisper                                                       |
| Compositing    | MoviePy (1.x) + FFmpeg                                               |
| Upload         | YouTube Data API v3 (`google-api-python-client`)                     |
| Deploy         | GitHub Actions (cron 2×/day + manual trigger with 12GB swap)         |

---

## AI-Coded Light Neon Manim Engine

For rich dynamic visuals (starting right from **second 0 in Scene 1 Hook**), [`src/ai_code_runner.py`](src/ai_code_runner.py) prompts the LLM to write complete, standalone Python animation code rendered via **Manim Community Edition**:

- **Light Neon Aesthetic**: Glowing lines, neon borders, pulses, and HUD elements on an ultra-dark background (`#05060f`) using curated cyberpunk palettes (Cyan `#00f0ff`, Pink `#ff007f`, Green `#00ff88`, Yellow `#ffe600`, Purple `#a855f7`, Orange `#ff7700`).
- **Zero LaTeX Requirement**: Uses custom `sym()` Monospace text instead of LaTeX `MathTex` or `Tex`. Runs reliably on minimal CI and local environments without `texlive` or `dvisvgm`.
- **Automatic AI Self-Repair**: If generated code fails with a syntax or runtime error during Manim rendering, `repair_code_with_ai()` feeds the error traceback back to the LLM to automatically patch and re-render.
- **Safe Frame Guarantee**: All objects are bounded to safe screen coordinates (`x ∈ [-6.2, 6.2], y ∈ [-3.3, 3.3]`) to prevent edge clipping on 16:9 displays.

---

## The `mathviz` Animation Engine

`src/mathviz/` is a lightweight, self-contained animation library inspired by
Manim but rendered entirely with **Pillow + NumPy**. This keeps CI fast and
dependency-light — no Cairo, GLSL, or LaTeX required.

**Core concepts**

- **Coordinate system** — pixel space (default 1920×1080), origin top-left, with
  2× supersampling for crisp anti-aliased edges (downscaled with LANCZOS at
  finalize).
- **Camera** — implemented as a coordinate transform at the `Canvas` primitive
  level (`_px(x, y)` applies pan + zoom). **Camera zoom is locked permanently to 1.0**
  to guarantee that text, diagrams, and formulas are never cropped at the screen edges.
- **Drawables** — `FunctionGraph`, `ParametricCurve`, `Circle`, `Rect`,
  `Polygon`, `Text`, `Dot`, `Line`, `Arrow`, `Axes`, `Formula`, `NeuralNet`,
  `BarChart`, `Group`. Many support a neon `glow` effect (layered translucent
  strokes).
- **Animations** — `FadeIn/Out`, `Write`, `DrawLine`, `GrowFromCenter`, `Move`,
  `CountUp`, `Pulse`, `Signal`, `CameraMove`, `Transform`, `MoveAlongPath`
  (a dot travels along a curve, optionally tracing it), and **`MorphShape`**.
- **Easing** — `linear`, `smooth`, `smoother`, `ease_in/out/in_out`, `back`,
  `elastic`, `bounce`.

**True vertex morphing (`MorphShape`)**

Unlike a simple cross-fade, `MorphShape` performs genuine point-set interpolation
(the classic 3Blue1Brown circle→square morph):

1. Sample outline point-sets from the source and destination shapes
   (`get_outline(n)` on `Circle`, `Rect`, and `Polygon`).
2. Rotate each point list so vertex 0 aligns near the same angular direction from
   the centroid — this minimizes twisting during the morph.
3. Linearly interpolate each vertex per frame into a `Polygon`'s live point buffer.

**Theming** — `theme.py` holds a `THEME` singleton (1920×1080 @ 30fps, GitHub-dark
palette `#0d1117` background, `#58a6ff` accent, 7-color palette, Be Vietnam Pro
fonts with full Vietnamese diacritics support).

---

## Multi-Tier Fallbacks

The pipeline is built to survive provider outages and quota limits. **You do not
need every key** — the minimum to run is **one working LLM key**; TTS works with
no key at all.

**LLM chain** (`config.yaml → llm.providers`, tried top to bottom):

| Tier | Provider      | Env var                                     | Notes                       |
| ---- | ------------- | ------------------------------------------- | --------------------------- |
| 0    | Anthropic     | `ANTHROPIC_API_KEY`                         | Claude Opus (primary)       |
| 1    | Gemini        | `GEMINI_API_KEY`                            | Gemini Flash                |
| 2    | Groq          | `GROQ_API_KEY`                              | gpt-oss-120b                |
| 3    | OpenRouter    | `OPENROUTER_API_KEY`                        | free Llama 3.3 70B          |
| 3.5  | Z.ai          | `ZAI_API_KEY`                               | GLM-4-Flash                 |
| 4    | Mistral       | `MISTRAL_API_KEY`                           | mistral-large               |
| 4.5  | NVIDIA NIM    | `NVIDIA_API_KEY`                            | Llama 3.3 70B               |
| 5    | GitHub Models | `GITHUB_MODELS_TOKEN`                       | gpt-4o-mini (free)          |
| 5    | SambaNova     | `SAMBANOVA_API_KEY`                         | Llama 3.3 70B (free)        |
| 5    | Cloudflare    | `CLOUDFLARE_API_TOKEN` + `CF_ACCOUNT_ID`    | Workers AI Llama 3.3 (free) |

Any provider missing its key is skipped automatically; the first one with a valid
key and a successful response wins.

**TTS chain** (`config.yaml → tts.providers`):

1. **VieNeu-TTS** — high-quality Vietnamese voice (installed from git in CI).
2. **Edge-TTS** — free Microsoft neural voices, no key required (final fallback).

(ElevenLabs is wired up but disabled by default.)

**Image search chain** (`src/image_fetcher.py`, tried top to bottom):

1. **Google Custom Search API** (`GOOGLE_API_KEY` + `GOOGLE_CSE_ID`) — High-precision search across curated domains (Pinterest, ByteByteGo, GeeksforGeeks, Wikimedia, Unsplash...).
2. **DuckDuckGo & Bing 16:9 Web Search** — **100% free, zero configuration, no key required**. Automatically fetches widescreen 16:9 HD images from the entire web if Google keys are absent or rate-limited.
3. **Pexels API** (`PEXELS_API_KEY`) — High-res stock photography and video b-roll.
4. **Openverse** — Creative Commons open-license image repository (keyless fallback).

---

## Local Setup

```bash
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# Linux / macOS
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

# Optional: high-quality Vietnamese TTS (falls back to Edge-TTS if skipped)
pip install git+https://github.com/pnnbao97/VieNeu-TTS.git

cp .env.example .env   # then fill in at least one LLM key
```

**FFmpeg is required.**

- Windows: `choco install ffmpeg` or download from <https://ffmpeg.org>
- macOS: `brew install ffmpeg`
- Ubuntu/Debian: `sudo apt-get install -y ffmpeg fonts-dejavu-core`

Python 3.11 is recommended (matches the CI runner).

---

## API Keys

Fill these into `.env` (see [`.env.example`](.env.example)). Only **one LLM key**
is strictly required.

| Variable                | Required?              | Where to get it                              |
| ----------------------- | ---------------------- | -------------------------------------------- |
| `GEMINI_API_KEY`        | one LLM key required   | <https://aistudio.google.com/apikey>         |
| `ANTHROPIC_API_KEY`     | one LLM key required   | your Anthropic-compatible provider           |
| `GROQ_API_KEY`          | optional LLM tier      | <https://console.groq.com>                   |
| `OPENROUTER_API_KEY`    | optional LLM tier      | <https://openrouter.ai>                      |
| `GOOGLE_API_KEY`        | optional image search  | <https://console.cloud.google.com/apis/credentials> |
| `GOOGLE_CSE_ID`         | optional image search  | <https://programmablesearchengine.google.com/> |
| `PEXELS_API_KEY`        | optional               | <https://www.pexels.com/api/> (nicer images + b-roll) |
| `HF_TOKEN`              | optional               | <https://huggingface.co/settings/tokens> (AI thumbnails) |
| `ELEVENLABS_API_KEY`    | optional               | <https://elevenlabs.io>                      |
| `YOUTUBE_CLIENT_SECRET` | only for uploading     | see [YouTube OAuth](#youtube-oauth-one-time) |
| `YOUTUBE_TOKEN`         | only for uploading     | see [YouTube OAuth](#youtube-oauth-one-time) |

If Google search keys are omitted, the pipeline automatically falls back to **DuckDuckGo & Bing web search** at zero cost with no keys required. Without `PEXELS_API_KEY`, images fall back to Openverse and video b-roll is disabled. Without the two YouTube variables, the pipeline still renders — it just won't upload.

---

## YouTube OAuth (one-time)

1. Go to <https://console.cloud.google.com> and create a project.
2. Enable **YouTube Data API v3**.
3. Create an **OAuth client ID** of type *Desktop app*, download it, and save it
   as `credentials/client_secret.json`.
4. Authenticate and generate a token:
   ```bash
   python -m src.youtube_uploader --auth
   ```
   A browser opens; sign in with the channel you want to publish to. The command
   prints a JSON token string.
5. For CI, put the **contents** of `client_secret.json` into the
   `YOUTUBE_CLIENT_SECRET` secret and the printed token into `YOUTUBE_TOKEN`
   (each as single-line JSON).

---

## Running Locally

```bash
# Print the generated script only (no rendering)
python -m src.pipeline --dry-run

# Render to output/ without uploading (good for testing)
python -m src.pipeline --no-upload

# Force a specific format
python -m src.pipeline --no-upload --mode long    # 16:9 explainer (~3-5 mins)
python -m src.pipeline --no-upload --mode short   # 9:16 Short (up to 3 mins / 180s)
python -m src.pipeline --no-upload --mode mega    # 16:9 Mega deep-dive (15-30+ mins)

# Run with a custom topic and target duration
python -m src.pipeline --no-upload --mode mega --duration 900 --topic "Kiến trúc Microservices và Event-Driven"

# Full run (renders and uploads)
python -m src.pipeline
```

The finished video is written to `output/video_<id>/video.mp4`.

---

## Deploying to GitHub Actions

The workflow [`.github/workflows/create-video.yml`](.github/workflows/create-video.yml)
handles the whole environment automatically — it configures Python 3.11, 12GB swap space,
FFmpeg, Manim, and all Python dependencies. You only need to provide secrets.

1. **Push the repo to GitHub.** `.env` is gitignored, so your keys stay local.

2. **Add repository secrets** — *Settings → Secrets and variables → Actions →
   New repository secret*. Add **at least one LLM key**; everything else is
   optional:

   | Secret                                                | Purpose                          |
   | ----------------------------------------------------- | -------------------------------- |
   | At least one of the LLM keys in the table above       | Script writing (any single tier) |
   | `GOOGLE_API_KEY`, `GOOGLE_CSE_ID`                     | Google Custom Search (falls back to DuckDuckGo/Bing) |
   | `PEXELS_API_KEY`                                       | Nicer images + video b-roll      |
   | `HF_TOKEN`                                             | AI-generated thumbnails          |
   | `YOUTUBE_CLIENT_SECRET`, `YOUTUBE_TOKEN`              | Auto-upload to YouTube           |

3. **Grant write permission** — *Settings → Actions → General → Workflow
   permissions → Read and write permissions*. The workflow commits
   `output/state.db` (topic-dedup history) back to the repo.

4. **Schedule (already configured):**
   - `00:00 UTC` (07:00 Vietnam) → long 16:9 explainer
   - `12:00 UTC` (19:00 Vietnam) → vertical 9:16 Short

5. **Manual run / Custom topic & duration** — *Actions → "Tạo & upload video tech" → Run
   workflow*. You can configure:
   - `mode`: `long`, `short`, or `mega` (15–30+ minute long-form video).
   - `topic`: Specify any custom topic (e.g. *Cơ chế hoạt động của Transformers & GPT*).
   - `duration`: Target duration in seconds (e.g. `900` for 15 mins, `1800` for 30 mins).
   - `no_upload`: Set to `true` to render without uploading (video is saved as an artifact for 3 days).

---

## Configuration Reference

All behavior is tunable in [`config.yaml`](config.yaml).

| Key                              | Description                                                        |
| -------------------------------- | ------------------------------------------------------------------ |
| `videos_per_run`                 | Number of videos generated per pipeline invocation.                |
| `language`                       | Content language: `vi` or `en`.                                    |
| `default_mode` / `modes`         | Video format presets: `long`, `short`, and `mega`.                 |
| `animation.ai_code_primary`      | Prioritize AI-generated Python/Manim animations ($\ge 70\%$).      |
| `animation.animate_from_start`   | Enforce animation from second 0 (Scene 1 Hook).                    |
| `llm.providers`                  | Ordered fallback chain (model, base_url, key env, retries).        |
| `llm.temperature` / `max_retries`| Global sampling temperature and retry budget.                      |
| `tts.providers`                  | TTS fallback order and per-provider voice settings.                |
| `visual.*`                       | Resolution, fps, colors, fonts, crossfade duration.                |
| `images.enabled`                 | Auto-fetch illustration images per scene.                          |
| `videos.enabled` / `max_per_video`| Auto-fetch Pexels video b-roll (needs `PEXELS_API_KEY`).          |
| `thumbnail.*`                    | AI thumbnail source, model, and Kurzgesagt style.                  |
| `subtitles.*`                    | Whisper model, burn-in toggle.                                     |
| `music` / `sfx` / `branding`     | Background music, transition SFX, intro/outro clips.               |
| `youtube.*`                      | Privacy status, category, default tags, made-for-kids flag.        |
| `topics.domains` / `dedup_days`  | Topic domains the LLM chooses from, and repeat-avoidance window.    |

To produce exactly two videos per day, keep `videos_per_run: 1` with the two cron
triggers, or set `videos_per_run: 2` with a single trigger.

---

## Project Layout

```
src/
├── pipeline.py          # Orchestrator (entry point: python -m src.pipeline)
├── topic_selector.py    # Picks a fresh topic (LLM + DB dedup)
├── script_writer.py     # LLM → structured Script (+ hierarchical mega-mode)
├── ai_code_runner.py    # AI Python/Manim animation generator & self-repair
├── llm.py               # Multi-tier LLM client with fallback
├── animation_bridge.py  # Safe (no eval/exec) bridge: Script → mathviz Scene
├── visual_engine.py     # Static scene + overlay rendering
├── compositor.py        # MoviePy/FFmpeg sequencing, music, SFX, crossfades
├── tts.py               # TTS with provider fallback + per-video voice lock
├── subtitles.py         # faster-whisper subtitles
├── image_fetcher.py     # Google / DuckDuckGo / Bing / Pexels / Openverse
├── thumbnail_ai.py      # AI thumbnail background (FLUX / Pollinations)
├── metadata.py          # YouTube title/description/tags + thumbnail overlay
├── youtube_uploader.py  # YouTube Data API v3 upload + --auth flow
├── models.py            # Dataclasses: Script, Scene, Exercise, ...
├── config.py            # Loads config.yaml, applies mode, exposes CONFIG
├── db.py                # SQLite state (topic history, video status)
└── mathviz/             # Self-written Pillow + NumPy animation engine
    ├── core.py          # Canvas primitives + Camera (zoom locked to 1.0)
    ├── objects.py       # Drawables (graphs, shapes, Polygon, NeuralNet, ...)
    ├── anims.py         # Animations (MorphShape, MoveAlongPath, Transform, ...)
    ├── custom_scene.py  # Declarative JSON → Scene interpreter
    ├── theme.py         # THEME singleton (palette, fonts, dimensions)
    └── easing.py        # Easing functions
```

---

## Security Model

- **No `eval`/`exec` of LLM output.** Math expressions go through an AST-validated
  `make_safe_fn` ([`src/animation_bridge.py`](src/animation_bridge.py)); the
  `custom` animation preset is **declarative JSON** interpreted by
  `custom_scene.py`, which enforces hard limits (max objects, steps, animations
  per step, run time, etc.).
- **Secrets stay out of git** — `.env` and `credentials/` are gitignored; CI reads
  everything from GitHub Actions Secrets.
- **Least dependencies** — no LaTeX/Cairo/GLSL; the animation engine is pure
  Python to keep the attack surface and CI footprint small.

---

## Troubleshooting

- **All LLM tiers fail** — check that at least one key in the chain is valid; the
  logs show which tier was tried and why it fell through.
- **No audio / robotic voice** — VieNeu-TTS may have failed to install; the
  pipeline falls back to Edge-TTS automatically (needs network access, no key).
- **Static images instead of video b-roll** — `PEXELS_API_KEY` is missing; b-roll
  auto-disables and static images are used.
- **Upload skipped** — `YOUTUBE_CLIENT_SECRET` / `YOUTUBE_TOKEN` not set, or the
  token expired (re-run `python -m src.youtube_uploader --auth`).
- **CI can't commit history** — enable *Read and write permissions* under
  *Settings → Actions → General*.
- **Duplicate topics** — increase `topics.dedup_days`; history lives in
  `output/state.db` (committed back by CI).

---

## Notes

- **YouTube quota** — each upload costs ~1,600 of 10,000 daily units, so two
  videos/day is comfortable.
- **Content policy** — automated content should provide genuine value; avoid
  patterns that trigger spam flags.
- **Assets** — drop background music into `assets/music`, transition SFX into
  `assets/sfx`, and fonts into `assets/fonts`.
