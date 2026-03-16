# ContextForge

> **Systematically test what's in your LLM context** — removes sections one at a time, measures quality impact via LLM-as-judge, and produces optimization recommendations with cost savings projections, all powered by Amazon Nova 2 Lite.

![Python](https://img.shields.io/badge/Python-3.12+-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Tests](https://img.shields.io/badge/Tests-161_passing-brightgreen)
![Nova 2 Lite](https://img.shields.io/badge/Nova_2_Lite-Bedrock_Converse-FF9900?logo=amazon&logoColor=white)
![Track](https://img.shields.io/badge/Track-Agentic_AI-orange)

**Amazon Nova AI Hackathon 2026** by Amazon & Devpost

---

## Table of Contents

- [Why This Matters](#why-this-matters-beyond-manual-prompt-trimming)
- [See It In Action](#see-it-in-action)
- [Architecture](#architecture)
- [Amazon Nova 2 Integration](#amazon-nova-2-integration-the-agentic-ai)
- [Quick Start](#quick-start)
- [Experiment Modes](#experiment-modes)
- [How It Works](#how-it-works-the-ablation-pipeline)
- [Features](#features)
- [Technical Stack](#technical-stack)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Future Work](#future-work)
- [License](#license)

---

## Why This Matters: Beyond Manual Prompt Trimming

| Manual Prompt Engineering | ContextForge |
|---------------------------|--------------|
| Guess which sections to remove | Systematically ablates every section and measures impact |
| No quality measurement | LLM-as-judge scoring across multiple reasoning tiers |
| One-off decisions | Reproducible experiments with statistical analysis |
| "Feels shorter" cost savings | Precise token reduction with dollar-amount projections |
| Trial and error | Pareto-optimal configurations computed automatically |
| No documentation | Publication-ready HTML reports with interactive charts |
| Generic advice | AI-generated Context Diet Plan with section-specific recommendations |

---

## See It In Action

When you run an ablation experiment, ContextForge walks through each phase automatically:

```
┌─────────────────────────────────────────────────────────────┐
│  CONTEXTFORGE — Ablation Experiment                         │
│  Mode: Quick  |  Sections: 7  |  Queries: 5  |  Tiers: 3  │
└─────────────────────────────────────────────────────────────┘

Phase 1: Parse & Segment
  ✓ Loaded 7 sections (212,438 tokens)
  ✓ Detected 12 redundant pairs (TF-IDF ≥ 0.70)

Phase 2: Baseline Evaluation
  ✓ Tier disabled — avg quality: 8.42 / 10
  ✓ Tier low      — avg quality: 8.65 / 10
  ✓ Tier high     — avg quality: 8.78 / 10

Phase 3: Single-Section Ablation
  ✓ system_prompt    Δ = 3.21  ★ ESSENTIAL
  ✓ faq_001          Δ = 0.84  ● MODERATE
  ✓ catalog_001      Δ = 0.12  ○ REMOVABLE
  ✓ conv_001–035     Δ = 0.03  ○ REMOVABLE
  ✓ conv_036–040     Δ = 2.45  ★ ESSENTIAL
  ✓ tool_001–018     Δ = 0.08  ○ REMOVABLE
  ✓ legal_001        Δ = 0.15  ○ REMOVABLE

Phase 4: Multi-Section Ablation
  ✓ Greedy elimination: removed 4 sections
  ✓ Quality retention: 97.2%
  ✓ Token reduction:   58.1% (123,416 tokens saved)

┌─────────────────────────────────────────────────────────────┐
│ RESULTS SUMMARY                                             │
├─────────────────────────────────────────────────────────────┤
│ Lean Config:  89,022 tokens (was 212,438)                   │
│ Quality:      97.2% retained                                │
│ Cost Savings: $0.037 per call → $37 per 1K calls            │
│ API Calls:    148  |  Experiment Cost: $0.07                 │
│                                                             │
│ Key Findings:                                               │
│   - Product catalog (100K tokens) is 47% of payload but     │
│     contributes only 0.12 quality points                    │
│   - Conversation turns 1–35 are fully removable (Δ ≈ 0)    │
│   - 18/20 tool definitions unused — safe to remove          │
│   - FAQ has 40% internal redundancy (12 duplicate pairs)    │
└─────────────────────────────────────────────────────────────┘
```

> A working demo of the Streamlit app can be viewed at [YouTube Demo](https://www.youtube.com/watch?v=PLACEHOLDER).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       CONTEXTFORGE                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                 ORCHESTRATION LAYER                    │ │
│  │ AblationEngine — main experiment loop & state machine │ │
│  │ Pydantic v2 state management — typed models           │ │
│  │ Mode selection — Demo / Quick / Full                   │ │
│  └────────────────────────────────────────────────────────┘ │
│                             │                               │
│                             ▼                               │
│  ┌────────────────────────────────────────────────────────┐ │
│  │           COGNITIVE CORE  (Nova 2 Lite)               │ │
│  │                                                        │ │
│  │  QualityScorer — LLM-as-judge scoring (MEDIUM tier)   │ │
│  │  DietPlanner — optimization recommendations (HIGH)    │ │
│  │  ReportGenerator — narrative HTML reports             │ │
│  │  QueryGenerator — auto-generates eval queries         │ │
│  │                                                        │ │
│  │  Extended Thinking across 4 tiers measures how        │ │
│  │  reasoning depth affects context sensitivity          │ │
│  └────────────────────────────────────────────────────────┘ │
│                             │                               │
│                             ▼                               │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                 EXECUTION LAYER                        │ │
│  │  ContextParser — JSON → Pydantic ContextPayload       │ │
│  │  Assembler — sections → Converse API parameters       │ │
│  │  Analyzer — local stats (numpy/scipy/sklearn)         │ │
│  │  RedundancyDetector — TF-IDF cosine similarity        │ │
│  │  VisualizationGenerator — Plotly interactive charts   │ │
│  └────────────────────────────────────────────────────────┘ │
│                             │                               │
│                             ▼                               │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                  PERSISTENCE LAYER                     │ │
│  │  Streamlit session state — experiment results cache    │ │
│  │  JSON artifacts — diet plans, reports, cost logs       │ │
│  │  HTML reports — self-contained downloadable files      │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Amazon Nova 2 Integration: The Agentic AI

This project leverages Amazon Nova 2 Lite's unique capabilities for autonomous context optimization.

### Four Cognitive Components

| Component | Role | Nova Feature | Output |
|-----------|------|-------------|--------|
| **QualityScorer** | LLM-as-judge — scores responses on 4 criteria | Extended Thinking (MEDIUM) | Structured JSON: scores per criterion + justification |
| **DietPlanner** | Generates optimization recommendations | Extended Thinking (HIGH) | Markdown diet plan with section-specific advice |
| **ReportGenerator** | Writes analytical narrative for HTML report | Code Interpreter (optional) | HTML report with Plotly charts + narrative |
| **QueryGenerator** | Auto-generates evaluation queries from context | Converse API | Structured JSON: queries with reference answers |

### Extended Thinking as a Research Variable

Unlike typical LLM applications that pick one reasoning tier, ContextForge uses all four tiers (disabled, low, medium, high) as an **experimental variable**. Each ablation is tested across tiers to measure how reasoning depth affects context sensitivity — a section that matters at `disabled` tier may be irrelevant when the model can reason through its absence.

- **disabled** — no reasoning, fastest baseline
- **low** — minimal chain-of-thought
- **medium** — standard reasoning (used for quality scoring)
- **high** — deep reasoning (used for diet plan generation; does NOT accept temperature/maxTokens parameters)

### Why This Qualifies for Agentic AI

- **Autonomous**: Runs 35–800+ API calls without human intervention across a multi-phase pipeline
- **Long-Running**: Full mode experiments run 15–30 minutes with hundreds of sequential decisions
- **Self-Correcting**: Retries with fallback reasoning tiers on parse failures; greedy elimination adapts to intermediate results
- **Explainable**: Every quality score includes per-criterion justification; diet plan explains *why* each section should be kept or removed

---

## Quick Start

### Prerequisites

- Python 3.12+
- An AWS account with [Amazon Bedrock](https://aws.amazon.com/bedrock/) access
- Model access enabled for `amazon.nova-2-lite-v1:0` in the Bedrock console (region: `us-east-1`)

### 1. Setup

```bash
git clone https://github.com/srikar161720/contextforge.git
cd contextforge

python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure AWS Credentials

```bash
cp .env.example .env
```

Edit `.env` and set your credentials:

```
AWS_ACCESS_KEY_ID=your_access_key_id
AWS_SECRET_ACCESS_KEY=your_secret_access_key
AWS_DEFAULT_REGION=us-east-1
```

### 3. Run the App

```bash
streamlit run app/main.py
# Open http://localhost:8501
```

### 4. Try the Demo Payload

Upload the included demo payload to see ContextForge in action:

```
data/demo_payloads/customer_support.json
```

This is a ~212K-token "bloated customer support agent" context designed to produce dramatic findings — ~58% of tokens removable with <3% quality loss.

### 5. Generate Your Own Demo Data

```bash
# Online — calls Nova API to generate realistic content
python scripts/generate_demo_payload.py

# Offline — uses templates, no API calls needed
python scripts/generate_demo_payload.py --offline
```

---

## Experiment Modes

| | Demo | Quick | Full |
|---|---|---|---|
| **Eval Queries** | 3 | 5 | 10 |
| **Reasoning Tiers** | disabled, medium | disabled, low, high | disabled, low, medium, high |
| **Repetitions** | 1 | 1 | 3 |
| **Single-Section Ablation** | Yes | Yes | Yes |
| **Multi-Section Ablation** | No | Yes | Yes |
| **Ordering Experiments** | No | No | Yes |
| **Est. API Calls** | ~35 | ~150 | ~800 |
| **Est. Duration** | 2–4 min | 5–10 min | 15–30 min |
| **Est. Cost** | ~$0.02 | ~$0.08 | ~$0.40 |

---

## How It Works: The Ablation Pipeline

```
Input: Context Payload (JSON) + Eval Queries + Mode Selection
                    │
                    ▼
            ┌────────────────┐
            │ PARSE & SEGMENT│  Validate JSON, count tokens, detect redundancy
            └───────┬────────┘
                    │
                    ▼
            ┌────────────────┐
            │    BASELINE    │  Score responses on full context across all tiers
            └───────┬────────┘
                    │
                    ▼
    ┌────────────────────────────────────┐
    │     SINGLE-SECTION ABLATION       │
    │                                    │
    │  For each section:                │
    │    1. Remove section from context │
    │    2. Run all queries per tier    │
    │    3. Score via LLM-as-judge      │
    │    4. Compute quality delta       │
    │    5. Classify impact             │
    │                                    │
    │  Output: ranked section impacts   │
    └───────────────┬────────────────────┘
                    │ (Quick + Full only)
                    ▼
    ┌────────────────────────────────────┐
    │     MULTI-SECTION ABLATION        │
    │                                    │
    │  Greedy backward elimination:     │
    │    1. Sort removable by tokens    │
    │    2. Remove one, re-evaluate     │
    │    3. Stop at quality tolerance   │
    │                                    │
    │  Output: lean configuration       │
    └───────────────┬────────────────────┘
                    │ (Full only)
                    ▼
    ┌────────────────────────────────────┐
    │     ORDERING EXPERIMENTS          │
    │                                    │
    │  Top 5 sections × 3 positions:   │
    │    start, middle, end             │
    │                                    │
    │  Output: position recommendations │
    └───────────────┬────────────────────┘
                    │
                    ▼
          ┌───────────────────┐
          │ ANALYSIS & REPORT │  Pareto frontier, confidence intervals,
          │                   │  diet plan, HTML report, cost projections
          └─────────┬─────────┘
                    │
                    ▼
Output: Dashboard + Diet Plan + HTML Report + Cost Savings
```

### Section Classification

| Classification | Avg Quality Delta | Meaning |
|----------------|-------------------|---------|
| Essential | >= 2.0 | Removing this section causes significant quality loss |
| Moderate | 0.5 – 2.0 | Noticeable impact — keep unless budget-constrained |
| Removable | 0 – 0.5 | Safe to remove with minimal quality impact |
| Harmful | < 0 | Removing this section *improves* quality |

---

## Features

### Supported Section Types

- **system_prompt** — Agent persona, rules, tone guidelines
- **rag_document** — Retrieved knowledge articles, FAQs, product catalogs
- **conversation_turn** — Chat history messages
- **tool_definition** — Function/API schemas for tool calling
- **few_shot_example** — Demonstration input/output pairs
- **custom** — Legal disclaimers, policies, or any other context

### Intelligent Ablation

- **Systematic**: Every section tested independently — no guesswork
- **Multi-tier**: Quality measured across reasoning depths to detect tier-sensitive sections
- **Adaptive**: Greedy elimination respects configurable quality tolerance (default 5%)
- **Statistical**: Confidence intervals and Pareto frontier analysis via numpy/scipy

### Rich Streamlit UI

Five-page interactive application with a pastel light theme:

1. **Upload & Configure** — Drop JSON payload, preview sections with token bars, select mode, tune thresholds
2. **Progress** — Live progress bar with background worker thread and fragment-based polling
3. **Results Dashboard** — 4 interactive Plotly charts, sortable detail table, redundancy clusters, lean config
4. **Context Diet Plan** — AI-generated optimization guide via Nova extended thinking (HIGH tier)
5. **Download Report** — Self-contained HTML report with dark-themed Plotly charts

### Interactive Charts

- **Section Impact Ranking** — Horizontal bar chart sorted by quality delta
- **Quality Heatmap** — Section x reasoning tier grid showing quality sensitivity
- **Tier Sensitivity Radar** — Spider chart of top 8 sections across tiers
- **Pareto Frontier** — Quality vs. token-count scatter plot with optimal configurations

### Report Generation

Nova generates a narrative HTML report via Jinja2 templating containing:

- Executive summary with key findings
- All 4 interactive Plotly charts (dark theme)
- Section-by-section analysis table
- Context Diet Plan with recommendations
- Cost projections and savings estimates
- Optional Code Interpreter analytical narrative (for demo flair)

### Redundancy Detection

TF-IDF cosine similarity analysis flags redundant section pairs above a configurable threshold (default 70%). Identifies duplicate content that inflates token count without adding unique information.

### Error Handling

- Graceful retry with exponential backoff for Bedrock API calls
- Fallback reasoning tiers on parse failures (medium → disabled)
- 4-strategy JSON parsing for LLM outputs (json → fence → repair → bracket)
- Code validation and timeout protection
- State preservation for experiment results

---

## Technical Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| LLM | Amazon Nova 2 Lite | Quality scoring, diet plans, reports, query generation |
| API | AWS Bedrock Converse | Structured API with system tools and extended thinking |
| Frontend | Streamlit | Multi-page interactive web application |
| Charts | Plotly | Interactive visualizations (pastel light + dark themes) |
| Validation | Pydantic v2 | Type-safe data models and input validation |
| Data | pandas, NumPy | Results tables and numerical computation |
| Statistics | scipy, scikit-learn | Confidence intervals, TF-IDF redundancy detection |
| Tokens | tiktoken | Pre-flight token count estimates (cl100k_base) |
| Reports | Jinja2 | HTML report template rendering |
| JSON Parsing | json-repair | Robust parsing of LLM-generated JSON |
| Config | python-dotenv, PyYAML | Environment variables and configuration management |

---

## Project Structure

```
contextforge/
├── app/
│   ├── main.py                        # Streamlit entry point — landing hero + CTA
│   ├── pages/
│   │   ├── 1_upload.py                # Upload JSON, preview sections, mode selection
│   │   ├── 2_progress.py             # Background thread + @st.fragment polling
│   │   ├── 3_results.py              # Dashboard: 4 charts, detail table, redundancy
│   │   ├── 4_diet_plan.py            # Diet plan generation + display (Nova HIGH)
│   │   └── 5_report.py               # HTML report generation + download + preview
│   └── components/
│       ├── layout.py                  # Shared layout: global CSS theme + sidebar
│       ├── context_viewer.py          # Section list with type/classification badges
│       ├── heatmap.py                 # Section × tier quality delta heatmap
│       ├── impact_chart.py            # Section impact horizontal bar chart
│       ├── pareto_chart.py            # Pareto frontier scatter chart
│       └── tier_radar.py             # Tier sensitivity radar chart
├── core/
│   ├── models.py                      # ALL Pydantic models — single source of truth
│   ├── parser.py                      # Context payload parser & segmenter
│   ├── assembler.py                   # Context assembly → Converse API parameters
│   ├── ablation_engine.py             # Ablation experiment orchestrator
│   ├── quality_scorer.py              # LLM-as-judge scoring via Nova
│   ├── analyzer.py                    # Local statistical analysis (numpy/scipy)
│   ├── redundancy.py                  # TF-IDF redundancy detection
│   ├── report_generator.py            # HTML report (Jinja2 + Plotly)
│   └── diet_planner.py               # Context Diet Plan via Nova (HIGH tier)
├── infra/
│   ├── bedrock_client.py              # Converse API wrapper (rate limiting, retry)
│   ├── rate_limiter.py                # Adaptive rate limiter (RPM + TPM)
│   ├── json_parser.py                 # 4-strategy JSON parsing
│   └── token_counter.py              # tiktoken pre-flight + API-reported actual
├── data/
│   ├── demo_payloads/                 # Pre-built demo payloads
│   │   └── customer_support.json      # Primary demo (~212K tokens)
│   ├── eval_queries/
│   │   └── customer_support_queries.json
│   └── templates/
│       └── report_template.html       # Jinja2 HTML report template
├── tests/                             # 161 tests (151 unit + 10 integration)
├── scripts/
│   ├── generate_demo_payload.py       # Generate demo data (online or offline)
│   ├── validate_demo.py               # Validate demo payload (26 checks)
│   └── cost_tracker.py               # Check API spend from cost_log.json
├── context/                           # Detailed reference docs
├── config.yaml                        # Model, pricing, rate limits, thresholds
├── requirements.txt
├── .env.example                       # AWS credential template
└── .streamlit/
    └── config.toml                    # maxUploadSize = 50
```

---

## Testing

```bash
# Run all tests (161 total)
python -m pytest tests/ -v

# Run unit tests only (151 — no AWS credentials needed)
python -m pytest tests/ -v -m "not integration and not slow"

# Run a specific component's tests
python -m pytest tests/test_parser.py -v

# Run integration tests (requires AWS credentials in .env)
python -m pytest tests/ -v -m integration
```

### Coverage

| Component | Tests |
|-----------|-------|
| ContextParser | 13 |
| Assembler | 12 |
| AblationEngine | 29 |
| Analyzer | 28 |
| QualityScorer | 16 |
| RedundancyDetector | 8 |
| JSONParser | 11 |
| DietPlanner | 16 |
| ReportGenerator | 21 |
| Day1Validation (integration) | 7 |
| **Total** | **161** |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'app'` | Run from project root: `streamlit run app/main.py` |
| `AWS credentials not found` | Ensure `.env` file exists with `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` |
| `AccessDeniedException` for Nova model | Enable `amazon.nova-2-lite-v1:0` in the Bedrock console (us-east-1) |
| Experiment timeout on large payloads | boto3 `read_timeout` is set to 3600s — extended thinking on large contexts is slow |
| Empty text output from Nova | `max_tokens` too low — reasoning tokens consume the output budget first |
| Quality scorer parse failure | Automatic retry with `reasoning_tier="disabled"` fallback |
| Streamlit upload fails | Check `.streamlit/config.toml` has `maxUploadSize = 50` |
| Rate limit errors (429) | Automatic retry with exponential backoff (RPM=200, TPM=8M) |
| Charts not rendering in report preview | Download the HTML file — interactive Plotly charts work best when opened directly |

---

## Future Work

- **Nova Embeddings** — Replace TF-IDF with `amazon.nova-2-multimodal-embeddings-v1:0` for high-precision redundancy detection
- **Web Grounding** — Enrich reports with external research via Nova's web grounding tool
- **Additional demo payloads** — Enterprise RAG and agentic workflow scenarios
- **Multi-model comparison** — Test ablation sensitivity across different Nova model sizes
- **Parallel execution** — Run independent ablation experiments concurrently
- **Export configurations** — Output lean configs as reusable JSON templates

---

## License

MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgments

- Built for the [Amazon Nova AI Hackathon 2026](https://amazonnovaaihackathon.devpost.com/) by Amazon & Devpost
- Powered by [Amazon Nova 2 Lite](https://aws.amazon.com/bedrock/) via AWS Bedrock Converse API
- Demo payload inspired by real-world customer support agent contexts

---

**Built with Amazon Nova 2 Lite for the Agentic AI Track** | **v0.1.0** | **161 Tests Passing**
