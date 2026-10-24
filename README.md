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
- [Series Engine & Auto-Curriculum Planning](#series-engine--auto-curriculum-planning)
- [Research Engine & 3-Tier Fact-Checking](#research-engine--3-tier-fact-checking)
- [AI Image & AI Video Generation](#ai-image--ai-video-generation)
- [Motion Graphics Engine (HTML5/Canvas & MathJax)](#motion-graphics-engine-html5canvas--mathjax)
- [StykAnimate 2D IK Stickman & Mascot Engine](#stykanimate-2d-ik-stickman--mascot-engine)
- [AI-Coded Light Neon Manim Engine](#ai-coded-light-neon-manim-engine)
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

- **End-to-end automation** — series planning → research → script → animation → TTS →
  subtitles → compositing → thumbnail → upload, with zero human input.
- **Research Engine with 3-Tier Fallback (Zero Hallucination)** — automatically researches
  and fact-checks topics via Wikipedia REST API / DuckDuckGo / ArXiv → Curated backup links
  (`assets/docs/sources.yaml`) → Bundled offline technical knowledge base (`assets/docs/*.md` & `.pdf`),
  enforcing 100% mathematical accuracy (e.g. RSA $5^7 \pmod{187} = 146$) and real-world trace data.
- **Series Engine & Auto-Curriculum Planning** — automatically plans multi-episode
  educational video series (3–7 episodes) when the database is empty or the previous
  series completes, threading narrative continuity hooks and `[Tập X/Y]` badges.
- **Motion Graphics Engine (HTML5/Canvas & MathJax)** — renders studio-grade dynamic scenes
  with smooth physics, binary trees, dynamic programming grids, and LaTeX-quality MathJax
  via headless Edge/Chrome.
- **StykAnimate 2D IK Stickman & Mascots** — integrates StykAnimate 2D Inverse Kinematics
  (Law of Cosines, FABRIK) with procedural actions (walk, run, pointing with neon laser, idea,
  coding, celebrate) and emotional Bunny Girl mascot reaction cards.
- **Dual Database Backend (Neon Serverless PostgreSQL + SQLite)** — seamlessly syncs
  state across cloud CI/CD runners (GitHub Actions) via `DATABASE_URL` with automatic
  zero-config fallback to local SQLite (`output/state.db`).
- **AI Image & Video Generation** — generates high-definition Kurzgesagt-style FLUX
  illustrations for scene backdrops (keyless, 100% free) plus 1 AI-generated cinematic
  motion video clip per video (`ai_motion` or Fal.ai diffusion).
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
  web search (100% free, no key needed) → Pexels → Openverse → AI FLUX illustration.
- **Resilient by design** — 10-tier LLM fallback, 3-tier research fallback, multi-tier TTS, and multi-tier
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
| Database       | Dual Backend: Neon Serverless PostgreSQL (`DATABASE_URL`) or SQLite Local (`output/state.db`) |
| Series Engine  | Autonomous curriculum planning (`src/series_manager.py`) with episode continuity hooks |
| Deep-ML AI     | 165 from-scratch verified ML/DL problems (`assets/docs/deepml/`) powering Long & Mega videos |
| onjsdev Shorts | 9:16 vertical algorithm & math visualizer (`src/onjsdev_visualizer.py`) with ASMR audio sync |
| Research Engine| 3-Tier Fact-Checking (`src/research_engine.py`): Wikipedia/DDG/ArXiv → Curated `sources.yaml` → Bundled offline KB (`assets/docs/`) |
| TTS            | VieNeu-TTS → Edge-TTS (free); ElevenLabs available but disabled by default |
| Motion Graphics| HTML5 Canvas + MathJax 3.x studio scenes via headless Edge/Chrome CLI (`src/motion_graphics.py`) |
| 2D Stickman/IK | StykAnimate 2D Inverse Kinematics (Law of Cosines, FABRIK) + Procedural Actions & Mascot Cards (`src/stick_animate.py`) |
| AI Animation   | Manim Community Edition (Light Neon / Cyberpunk style, zero LaTeX) with AI self-repair |
| Math Animation | Self-written `mathviz` (Pillow + NumPy, zoom locked to 1.0)           |
| Static visuals | Matplotlib (mathtext formulas, pure Python — no LaTeX)              |
| Images         | Google CSE → DuckDuckGo / Bing 16:9 → Pexels → Openverse → AI FLUX Illustration |
| Video b-roll   | Pexels Videos + AI Video Generator (`src/ai_video.py`: `ai_motion` / Fal.ai / Pollinations) |
| Thumbnail      | HuggingFace FLUX.1-schnell → Pollinations Flux (keyless fallback)   |
| Subtitles      | faster-whisper                                                       |
| Compositing    | MoviePy (1.x & 2.x via `mv_compat`) + FFmpeg                         |
| Upload         | YouTube Data API v3 (`google-api-python-client`)                     |
| Deploy         | GitHub Actions (cron 2×/day + manual trigger with 12GB swap)         |

---

## Series Engine & Auto-Curriculum Planning

Rather than generating isolated, one-off videos, the pipeline features an autonomous **Series Engine** ([`src/series_manager.py`](src/series_manager.py)) designed to boost YouTube audience retention and binge-watching.

- **Auto-Curriculum Planning**: Whenever the database is empty or the active series completes all its episodes, the pipeline automatically prompts the LLM to design a cohesive **3–7 episode educational curriculum** (e.g. *Bí Mật Của Hệ Thống Phân Tán*, *Kiến Trúc CPU & Bộ Nhớ Từ Số 0*).
- **Pedagogical Scaffolding**: Episodes progress logically from core intuitive concepts (Episode 1) to intermediate mechanics and real-world architectures (Final Episode).
- **Narrative Continuity**:
  - **Hook Opening (Scene 1)**: Automatically references the key takeaway from the previous episode (`hook_from_previous`).
  - **Outro Ending**: Teases the mystery and challenge to be solved in the upcoming episode (`hook_to_next`).
  - **YouTube Title**: Automatically prepends `[Tập X/Y]` to titles for clear binge-watching cues.
- **Dual Database Backend (Neon Serverless PostgreSQL + SQLite)**:
  - **Neon Cloud**: Set `DATABASE_URL=postgresql://...` in `.env` to synchronize series status across local development machines and GitHub Actions runners.
  - **SQLite Local**: If `DATABASE_URL` is omitted, the pipeline operates 100% offline using `output/state.db`.

```bash
# View active and completed series roadmap & episode status
python -m src.pipeline --list-series

# Actively plan a new series with a specific theme and episode count
python -m src.pipeline --plan-series "Kiến trúc Microservices & Event-Driven" --episodes 5

# Run the pipeline (automatically picks the next pending episode in the active series)
python -m src.pipeline
```

---

## Research Engine & 3-Tier Fact-Checking

To prevent LLM hallucinations on mathematical proofs, algorithm traces, and technical specifications, the pipeline features an autonomous **Research Engine** ([`src/research_engine.py`](src/research_engine.py)). Before writing any script or generating animation code, the system gathers verified context using a strict **3-Tier Fallback Hierarchy**:

```
[Topic Query]
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│ Tier 1: Auto Search & Download                              │
│ • Wikipedia REST & Action API (VI + EN, relevance-filtered) │
│ • DuckDuckGo Instant Answer API                             │
│ • ArXiv API (for ML & Distributed Systems papers)           │
│ • TinyFish Web Search Fallback                              │
└──────────────────────────────┬──────────────────────────────┘
                               │ (if failed or low relevance)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ Tier 2: Curated Backup Links (`assets/docs/sources.yaml`)   │
│ • Categorized across 8 domains (DSA, Crypto, AI, OS, etc.)  │
│ • Direct fetches from canonical docs (GeeksforGeeks, RFCs)  │
└──────────────────────────────┬──────────────────────────────┘
                               │ (if offline or timeout)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ Tier 3: Bundled Offline Technical Knowledge Base            │
│ • 8 verified Markdown manuals in `assets/docs/*.md`         │
│ • Concrete algorithm traces, exact formulas, Big-O tables   │
│ • Local `.pdf` documentation parsing via `pypdf`            │
└─────────────────────────────────────────────────────────────┘
```

### 1. Zero-Hallucination Pipeline Enforcement ("Làm Theo Tài Liệu")
- **Script Generation ([`src/script_writer.py`](src/script_writer.py))**: Research findings are injected into the LLM system prompt inside `<research>` tags. The LLM is strictly instructed to use the verified formulas, specific trace inputs/outputs (e.g. RSA $5^7 \pmod{187} = 146$, Two Pointers array `[1, 2, 4, 7, 11]`, target $13$), and accurate Big-O complexities in both spoken narration and visual prompts.
- **Visual & Animation Code ([`src/ai_code_runner.py`](src/ai_code_runner.py))**: Matplotlib and Manim code generation prompts ingest the research data directly. Graphs plot accurate numerical curves and animations reflect verified step-by-step state transitions rather than fabricated numbers.

### 2. High-Speed Research Caching
- Research results are normalized, hashed, and cached in `assets/cache/research/` (JSON format with configurable TTL).
- Subsequent runs on related topics retrieve verified facts in $< 50\text{ms}$ with zero redundant network requests.

---

## AI Image & AI Video Generation

### 1. Keyless AI Scene Illustration (`src/image_fetcher.py`)
Abstract computer science concepts (e.g. *Cache Coherence*, *Transformer Attention*, *Zero-Knowledge Proofs*) rarely match generic stock photos. The pipeline integrates an **AI Illustration Generator**:
- **FLUX.1-schnell via Pollinations AI**: 100% free, zero configuration, no API key required.
- **Consistent Tech Editorial Aesthetic**: Prompts are auto-engineered into flat vector Kurzgesagt / Cyberpunk style, strictly suppressing random text, watermarks, and logos.
- **Fallback Hierarchy**: Google CSE → DuckDuckGo/Bing → Pexels → Openverse → AI FLUX Illustration. If stock search yields nothing, the AI generator draws a tailored 1080p illustration.

### 2. Autonomous AI Video Generation (`src/ai_video.py`)
Each video can feature up to 1 AI-generated video clip (`config.yaml → ai_video.max_per_run: 1`):
- **Provider `ai_motion` (Default & 100% Free)**: Generates a high-resolution FLUX image and instantly renders a 5-second cinematic motion clip with smooth camera zoom, 3D pan, and cyber ambiance in ~2–3 seconds with zero credit costs.
- **Provider `fal` (Diffusion Models)**: Connects to Fal.ai API (`FAL_KEY`) to generate real video diffusion clips using state-of-the-art models like **Wan 2.1**, **Kling**, or **LTX-Video**.
- **Resilient Fallback**: If cloud video generation encounters rate limits or errors, it gracefully falls back to `ai_motion` to guarantee pipeline completion.

---

## Motion Graphics Engine (HTML5/Canvas & MathJax)

For studio-quality computational layouts and complex algorithm representations, [`src/motion_graphics.py`](src/motion_graphics.py) renders high-definition scenes using **HTML5 Canvas 2D + MathJax 3.x** through headless Microsoft Edge or Google Chrome:

- **Algorithmic & Data Structure Canvas Visualizations**:
  - **Dynamic Programming Tables**: Multi-colored matrix cells with active row/column backtracking highlights and score updates.
  - **Binary Trees & Graphs**: Auto-balanced node coordinates with animated edge pulses, BFS/DFS visitation colors, and pointer tags.
  - **Sorting & Array Animations**: Distinct colored blocks, dual comparison markers ($L, R, mid$), and swap trails.
  - **Neural Network Architectures**: Interactive input, hidden, and output layers with active synaptic weights and activation heatmaps.
- **Native LaTeX Formulas via MathJax 3.x**: Math formulas (e.g. $\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$, $\sigma(Wx+b)$, RSA modular equations) are rendered with pristine vector typography directly into the canvas without requiring bulky system LaTeX (`texlive`) installations.
- **Dynamic Contextual HUD**: Auto-renders dynamic chapter breadcrumbs (`scene.heading`) and progress tracks in the upper HUD so viewers always know their position in the curriculum.
- **Zero-Dependency Headless Rendering**: Utilizes standard Edge or Chrome CLI flags (`--headless`, `--screenshot`) discovered dynamically by [`src/asset_manager.py`](src/asset_manager.py) across Windows and Linux environments.

---

## StykAnimate 2D IK Stickman & Mascot Engine

To maximize viewer engagement and provide relatable pedagogical cues, [`src/stick_animate.py`](src/stick_animate.py) introduces a lightweight 2D skeletal animation engine:

- **2D Inverse Kinematics (IK)**:
  - Solves two-bone joint angles (thigh–shin, bicep–forearm) in closed-form using the **Law of Cosines** with reach clamps.
  - Employs **FABRIK** (Forward And Backward Reaching Inverse Kinematics) for multi-segment chains, ensuring natural poses without joint flipping or jitter.
- **Procedural Action Loops**:
  - `walk` / `run`: Full sinusoidal gait cycles with foot-strike, passing phase, and counter-balancing arm swings.
  - `point_laser`: Stickman aims a high-visibility neon laser beam at key code blocks or math formulas on screen.
  - `idea`: Sudden realization gesture with lightbulb glow, hand gesture, and energetic bounce.
  - `coding`: Realistic keyboard typing posture at a futuristic glowing workstation.
  - `celebrate`: Two-arm raised victory jump celebrating correct solutions and algorithmic insights.
- **Bunny Girl Emotional Mascot Cards**:
  - Rendered as discreet, expressive reaction overlays in the screen corner (e.g., *thinking*, *shocked at $O(n^2)$ complexity*, *proud of $O(n \log n)$ optimization*).
  - Enhances educational warmth and personalizes technical deep-dives.

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
5. **AI FLUX Illustration** (`images.ai_enabled: true`) — 100% free keyless FLUX illustration generation (Pollinations AI / HuggingFace) when stock images return no results or `images.ai_primary: true`.

**Database resilience chain** (`src/db.py`, 4-tier automatic fallback):

1. **Neon Serverless PostgreSQL** (`DATABASE_URL`) — Primary cloud persistence. Syncs multi-episode series state across local dev machines and GitHub Actions runners.
2. **Local SQLite** (`output/state.db`) — Automatic zero-downtime fallback if Neon connection drops, network times out, credentials are invalid, or Neon database is paused/suspended.
3. **In-Memory SQLite** (`:memory:`) — Automatic fallback if local file system is read-only, permissions fail, or SQLite file is locked by another process.
4. **Append-Only File Backup & Safe Execution** (`output/topics_history.txt`) — Topic history is always written to plain text; all DB writes are safe-wrapped with timestamp fallback IDs so video rendering and YouTube upload **NEVER crash** due to database errors.

**Research & Fact-Checking chain** (`src/research_engine.py`, 3-tier automatic fallback):

1. **Tier 1 (Automated Web/API Search)** — Wikipedia REST & Action API (Vietnamese & English with keyword relevance matching) → DuckDuckGo Instant Answer API → ArXiv API → TinyFish.
2. **Tier 2 (Curated Backup Links)** — Authoritative domain documentation specified in `assets/docs/sources.yaml` (8 technical categories).
3. **Tier 3 (Bundled Offline Technical Knowledge Base)** — 8 comprehensive verified Markdown docs (`assets/docs/*.md`) + local technical `.pdf` parsing via `pypdf`.

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
| `DATABASE_URL`          | optional (dual DB)     | <https://neon.tech> (PostgreSQL; fallback: local SQLite `output/state.db`) |
| `POLLINATIONS_API_KEY`  | optional (AI images)   | <https://pollinations.ai> (unlimited rate limits, free) |
| `FAL_KEY`               | optional (AI video)    | <https://fal.ai> (for Wan 2.1 / Kling diffusion video clips) |
| `GOOGLE_API_KEY`        | optional image search  | <https://console.cloud.google.com/apis/credentials> |
| `GOOGLE_CSE_ID`         | optional image search  | <https://programmablesearchengine.google.com/> |
| `PEXELS_API_KEY`        | optional               | <https://www.pexels.com/api/> (stock images + b-roll) |
| `HF_TOKEN`              | optional               | <https://huggingface.co/settings/tokens> (AI thumbnails) |
| `ELEVENLABS_API_KEY`    | optional               | <https://elevenlabs.io>                      |
| `YOUTUBE_CLIENT_SECRET` | only for uploading     | see [YouTube OAuth](#youtube-oauth-one-time) |
| `YOUTUBE_TOKEN`         | only for uploading     | see [YouTube OAuth](#youtube-oauth-one-time) |

If `DATABASE_URL` is set, the pipeline automatically connects to **Neon Serverless PostgreSQL** and keeps series/dedup state synced across local dev and GitHub Actions runners. If omitted, it automatically falls back to local SQLite (`output/state.db`).
AI images use **Pollinations FLUX** for 100% free, keyless generation (`POLLINATIONS_API_KEY` is optional for higher rate limits). AI video uses `ai_motion` by default (0-cost cinematic motion MP4 from FLUX images); setting `FAL_KEY` enables cutting-edge video diffusion models.
If Google search keys are omitted, the pipeline automatically falls back to **DuckDuckGo & Bing web search** at zero cost with no keys required. Without `PEXELS_API_KEY`, images fall back to Openverse/FLUX and stock video b-roll is skipped. Without the two YouTube variables, the pipeline still renders — it just won't upload.

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
# === 1. Series Engine & Curriculum Management ===
# View active/completed series roadmaps, episode progress, and status
python -m src.pipeline --list-series

# Autonomously plan a new curriculum series (3-7 episodes)
python -m src.pipeline --plan-series "Kiến Trúc Microservices & Event-Driven" --episodes 5

# Auto-run series: picks the next pending episode in the active series
# (If DB is empty or previous series completed, automatically plans a new series first!)
python -m src.pipeline --no-upload

# === 2. One-Off & Custom Topic Runs ===
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

# Full run (renders and uploads to YouTube)
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
   | `DATABASE_URL`                                        | Neon PostgreSQL connection string (syncs series across CI runs) |
   | `POLLINATIONS_API_KEY`                                | Pollinations AI key (higher rate limits for AI images) |
   | `FAL_KEY`                                             | Fal.ai key for AI diffusion video clips (Wan 2.1 / Kling) |
   | `GOOGLE_API_KEY`, `GOOGLE_CSE_ID`                     | Google Custom Search (falls back to DuckDuckGo/Bing) |
   | `PEXELS_API_KEY`                                       | Nicer images + video b-roll      |
   | `HF_TOKEN`                                             | AI-generated thumbnails          |
   | `YOUTUBE_CLIENT_SECRET`, `YOUTUBE_TOKEN`              | Auto-upload to YouTube           |

3. **Grant write permission** — *Settings → Actions → General → Workflow
   permissions → Read and write permissions*. If using local SQLite, the workflow commits
   `output/state.db` back to the repo. (With `DATABASE_URL`, state is synced directly to Neon).

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
| `series.auto_plan`               | Automatically plan a new 3–7 episode series when DB is empty/done.  |
| `series.default_episodes_min`    | Minimum episodes for auto-planned curriculum (default: `3`).       |
| `series.default_episodes_max`    | Maximum episodes for auto-planned curriculum (default: `7`).       |
| `research.enabled`               | Enable 3-tier fact-checking & research engine before scripting.    |
| `research.auto_download`         | Auto-search Wikipedia REST, DuckDuckGo & ArXiv (default: `true`).  |
| `research.use_backup_links`      | Auto-fetch from curated `assets/docs/sources.yaml` when tier 1 fails.|
| `research.use_bundled_docs`      | Auto-read bundled offline markdown docs in `assets/docs/*.md`.     |
| `research.cache_enabled`         | Cache queries in `assets/cache/research/` to prevent latency.      |
| `research.timeout`               | Network timeout in seconds for web research lookups (default: `10`).|
| `motion_graphics.enabled`        | Enable HTML5/Canvas & MathJax studio rendering via headless browser.|
| `stick_animate.enabled`          | Enable 2D Inverse Kinematics stickman procedural action loops.     |
| `stick_animate.mascot`           | Enable emotional Bunny Girl reaction mascot overlay cards.         |
| `animation.ai_code_primary`      | Prioritize AI-generated Python/Manim animations ($\ge 70\%$).      |
| `animation.animate_from_start`   | Enforce animation from second 0 (Scene 1 Hook).                    |
| `llm.providers`                  | Ordered fallback chain (model, base_url, key env, retries).        |
| `llm.temperature` / `max_retries`| Global sampling temperature and retry budget.                      |
| `tts.providers`                  | TTS fallback order and per-provider voice settings.                |
| `visual.*`                       | Resolution, fps, colors, fonts, crossfade duration.                |
| `images.enabled`                 | Auto-fetch illustration images per scene.                          |
| `images.ai_enabled`              | Fallback to AI FLUX illustration when stock search fails.          |
| `images.ai_primary`              | Prioritize AI FLUX illustrations over stock images for all scenes. |
| `images.ai_provider`             | AI image provider (`pollinations` free, `huggingface`).            |
| `images.ai_style`                | Style prompt injected into AI image generation.                    |
| `ai_video.enabled`               | Enable 1 AI cinematic video clip per generated video.               |
| `ai_video.max_per_run`           | Strict cap on AI video clips per video run (default: `1`).          |
| `ai_video.provider`              | AI video provider (`ai_motion` free default, `fal`, `pollinations`).|
| `ai_video.duration`              | Length in seconds of generated AI video clip (default: `5.0`).     |
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
├── series_manager.py    # Autonomous curriculum planner & series DB manager
├── research_engine.py   # 3-tier fact-checking & research engine (Web → Links → Offline KB)
├── topic_selector.py    # Picks a fresh topic (LLM + DB dedup)
├── script_writer.py     # LLM → structured Script (+ research context & series hooks)
├── ai_code_runner.py    # AI Python/Manim animation generator & self-repair (+ research context)
├── ai_video.py          # Autonomous 1-clip AI video generator (ai_motion / fal)
├── motion_graphics.py   # HTML5/Canvas & MathJax renderer via headless Edge/Chrome CLI
├── stick_animate.py     # StykAnimate 2D IK stickman procedural actions & mascot cards
├── onjsdev_visualizer.py# @onjsdev 9:16 Shorts engine (neon bars, code cards, ASMR sync)
├── sfx_generator.py     # Sound effects synthesizer (click, pop, swap tones, whoosh)
├── asset_manager.py     # Browser discovery (Edge/Chrome), fonts & asset downloader
├── scrapers/            # Web & problem scrapers
│   └── deepml_scraper.py# Downloads verified problems & code from deep-ml.com
├── mv_compat.py         # MoviePy 1.x & 2.x cross-version compatibility adapter
├── llm.py               # Multi-tier LLM client with fallback & quota handling
├── animation_bridge.py  # Safe (no eval/exec) bridge: Script → mathviz Scene
├── visual_engine.py     # Static scene + overlay rendering
├── compositor.py        # MoviePy/FFmpeg sequencing, music, SFX, crossfades
├── tts.py               # TTS with provider fallback + per-video voice lock
├── subtitles.py         # faster-whisper subtitles
├── image_fetcher.py     # Multi-tier image fetcher + keyless AI FLUX generator
├── thumbnail_ai.py      # AI thumbnail background (FLUX / Pollinations)
├── metadata.py          # YouTube title/description/tags + series badges
├── youtube_uploader.py  # YouTube Data API v3 upload + --auth flow
├── models.py            # Dataclasses: Script, Scene, Series, Episode, ...
├── config.py            # Loads config.yaml, applies mode, exposes CONFIG
├── db.py                # Dual-backend state: Neon PostgreSQL & SQLite
└── mathviz/             # Self-written Pillow + NumPy animation engine
    ├── core.py          # Canvas primitives + Camera (zoom locked to 1.0)
    ├── objects.py       # Drawables (graphs, shapes, Polygon, NeuralNet, ...)
    ├── anims.py         # Animations (MorphShape, MoveAlongPath, Transform, ...)
    ├── custom_scene.py  # Declarative JSON → Scene interpreter
    ├── theme.py         # THEME singleton (palette, fonts, dimensions)
    └── easing.py        # Easing functions

assets/docs/             # Offline technical knowledge base & sources
├── sources.yaml         # Curated authoritative links across 8 technical domains
├── deepml/              # 165 verified Machine Learning & Deep Learning from-scratch problems
│   ├── problems_index.json # Master searchable index of all 165 Deep-ML problems
│   ├── deep_learning.md    # Transformer, Attention, CNN, Backpropagation, MLP (48 problems)
│   ├── machine_learning_fundamentals.md # Gradient Descent, Regression, SVM, K-Means (54 problems)
│   └── linear_algebra.md   # Matrix operations, SVD, Eigenvalues, Covariance (33 problems)
├── algorithms_dsa.md    # DSA verified traces (Two Pointers, Quicksort, Dijkstra, Big-O)
├── cryptography_security.md # Crypto math (RSA modular steps, AES-GCM, Diffie-Hellman)
└── *.md                 # Computer Architecture, Graphics, Networking...
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
