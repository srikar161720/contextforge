"""
app/pages/1_upload.py — Context payload upload and experiment configuration.

Allows the user to:
  1. Upload a JSON context payload file.
  2. Preview the parsed sections via the context_viewer component.
  3. Select experiment mode (Demo / Quick / Full) with cost/duration guidance.
  4. Tune quality tolerance and redundancy threshold.
  5. Proceed to the progress page.

Session state written:
  - payload          (ContextPayload)
  - payload_name     (str)
  - experiment_config (ExperimentConfig)
"""

from __future__ import annotations

import json
from collections import Counter

import streamlit as st

from app.components.context_viewer import render_context_viewer
from core.models import ExperimentConfig, ExperimentMode
from core.parser import parse_payload

# ── Mode metadata shown in the UI ─────────────────────────────────────────────

_MODE_INFO: dict[str, dict] = {
    "demo": {
        "label":       "Demo",
        "emoji":       "⚡",
        "queries":     3,
        "tiers":       ["disabled", "medium"],
        "reps":        1,
        "api_calls":   "~35",
        "duration":    "2–4 min",
        "cost":        "~$0.02",
        "description": "Fast single-section sweep. No multi-section or ordering experiments. Best for demos and quick exploration.",
        "color":       "#6ee7b7",
    },
    "quick": {
        "label":       "Quick",
        "emoji":       "🚀",
        "queries":     5,
        "tiers":       ["disabled", "low", "high"],
        "reps":        1,
        "api_calls":   "~150",
        "duration":    "5–10 min",
        "cost":        "~$0.08",
        "description": "Single + multi-section ablation across 3 tiers. Includes greedy backward elimination for lean config.",
        "color":       "#fde68a",
    },
    "full": {
        "label":       "Full",
        "emoji":       "🔬",
        "queries":     10,
        "tiers":       ["disabled", "low", "medium", "high"],
        "reps":        3,
        "api_calls":   "~800",
        "duration":    "15–30 min",
        "cost":        "~$0.40",
        "description": "Comprehensive — all 4 tiers, 3 repetitions, multi-section elimination, and ordering experiments.",
        "color":       "#fca5a5",
    },
}


def _section_type_counter(payload) -> dict[str, int]:
    """Count sections by type."""
    counts: Counter = Counter(s.section_type.value for s in payload.sections)
    return dict(counts)


def _render_payload_summary(payload) -> None:
    """Render a 4-column summary card row for the parsed payload."""
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Sections", len(payload.sections))
    with col2:
        st.metric("Total Tokens", f"{payload.total_tokens:,}")
    with col3:
        st.metric("Eval Queries", len(payload.evaluation_queries))
    with col4:
        est_cost = payload.total_tokens * 0.30 / 1_000_000
        st.metric("Cost / Call (input)", f"${est_cost:.4f}")


def _render_mode_cards(selected_mode: str) -> None:
    """Render three mode info cards highlighting the selected mode."""
    cols = st.columns(3)
    for col, (mode_key, info) in zip(cols, _MODE_INFO.items()):
        with col:
            is_selected = mode_key == selected_mode
            border      = f"3px solid {info['color']}" if is_selected else "1px solid #e5e7eb"
            bg          = "#fafafa" if not is_selected else f"{info['color']}18"
            st.markdown(
                f'<div style="border:{border}; border-radius:12px; padding:1rem; '
                f'background:{bg}; height:100%;">'
                f'<div style="font-size:1.4rem;">{info["emoji"]}</div>'
                f'<div style="font-weight:700; color:#1f2937; margin:4px 0 2px;">'
                f'{info["label"]} Mode</div>'
                f'<div style="font-size:0.78rem; color:#6b7280; margin-bottom:8px;">'
                f'{info["description"]}</div>'
                f'<div style="font-size:0.78rem; color:#374151;">'
                f"⏱ {info['duration']} &nbsp;|&nbsp; "
                f"📞 {info['api_calls']} calls &nbsp;|&nbsp; "
                f"💰 {info['cost']}"
                f"</div>"
                f"</div>",
                unsafe_allow_html=True,
            )


# ── Page ──────────────────────────────────────────────────────────────────────

st.title("🔬 Upload & Configure")
st.markdown(
    "Upload your context payload JSON, preview the sections, choose an experiment "
    "mode, and start the ablation sweep."
)

st.markdown("---")

# ── File upload ───────────────────────────────────────────────────────────────

st.subheader("1. Upload Context Payload")
st.markdown(
    "Upload a JSON file following the ContextForge payload schema. "
    "You can use `data/demo_payloads/customer_support.json` to try a pre-built demo."
)

uploaded_file = st.file_uploader(
    "Drop your JSON payload here",
    type=["json"],
    help="Max file size: 50 MB. File must conform to the ContextForge payload schema.",
)

payload = None
if uploaded_file is not None:
    try:
        raw_bytes = uploaded_file.read()
        raw_dict  = json.loads(raw_bytes)
        payload   = parse_payload(raw_dict)
        st.session_state["payload"]      = payload
        st.session_state["payload_name"] = uploaded_file.name.removesuffix(".json")
        st.success(
            f"✅ Parsed **{len(payload.sections)} sections** "
            f"({payload.total_tokens:,} tokens) with "
            f"**{len(payload.evaluation_queries)} eval queries**."
        )
    except json.JSONDecodeError as exc:
        st.error(f"Invalid JSON — could not parse file: {exc}")
    except ValueError as exc:
        st.error(f"Payload validation failed: {exc}")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Unexpected error while parsing payload: {exc}")
elif "payload" in st.session_state:
    # Keep previously uploaded payload across reruns
    payload = st.session_state["payload"]
    st.info(
        f"Using previously loaded payload: **{st.session_state.get('payload_name', 'unknown')}** "
        f"({payload.total_tokens:,} tokens, {len(payload.sections)} sections)."
    )

# ── Payload preview ───────────────────────────────────────────────────────────

if payload is not None:
    st.markdown("---")
    st.subheader("2. Payload Preview")
    _render_payload_summary(payload)

    with st.expander("📋 Eval Queries", expanded=False):
        for i, eq in enumerate(payload.evaluation_queries, 1):
            st.markdown(
                f'<div style="padding:0.5rem 0.75rem; margin-bottom:6px; '
                f'background:#f8f9fc; border-radius:8px; border-left:3px solid #c4b5fd;">'
                f'<span style="font-size:0.78rem; color:#9ca3af;">Q{i}</span><br>'
                f'<span style="font-size:0.88rem; color:#374151;">{eq.query}</span>'
                f"</div>",
                unsafe_allow_html=True,
            )

    with st.expander("📂 Context Sections", expanded=True):
        render_context_viewer(payload)

# ── Experiment configuration ──────────────────────────────────────────────────

st.markdown("---")
st.subheader("3. Configure Experiment")

mode_options = ["Demo ⚡", "Quick 🚀", "Full 🔬"]
mode_keys    = ["demo", "quick", "full"]

selected_label = st.selectbox(
    "Experiment Mode",
    options=mode_options,
    index=0,
    help="Demo is fastest; Full is most comprehensive.",
)
selected_mode = mode_keys[mode_options.index(selected_label)]

_render_mode_cards(selected_mode)

st.markdown("")

with st.expander("⚙️ Advanced Settings", expanded=False):
    quality_tolerance = st.slider(
        "Quality Tolerance",
        min_value=0.01,
        max_value=0.20,
        value=0.05,
        step=0.01,
        format="%.0f%%",
        help=(
            "Maximum acceptable quality loss fraction when building the lean configuration. "
            "5% means the lean config must retain at least 95% of baseline quality."
        ),
    )
    redundancy_threshold = st.slider(
        "Redundancy Detection Threshold",
        min_value=0.50,
        max_value=0.95,
        value=0.70,
        step=0.05,
        format="%.2f",
        help=(
            "TF-IDF cosine similarity cutoff for flagging redundant section pairs. "
            "Lower = more pairs detected; higher = only very similar pairs flagged."
        ),
    )
else:
    quality_tolerance    = 0.05
    redundancy_threshold = 0.70

# ── Start experiment button ───────────────────────────────────────────────────

st.markdown("---")

if payload is None:
    st.info("⬆️ Upload a context payload to enable the experiment controls.")
else:
    mode_info = _MODE_INFO[selected_mode]
    st.markdown(
        f'<div style="background:#faf5ff; border:1px solid #c4b5fd; border-radius:10px; '
        f'padding:0.85rem 1.1rem; margin-bottom:1rem;">'
        f'<span style="font-weight:600; color:#7c6fea;">Ready to run:</span> '
        f'{mode_info["label"]} mode — {len(payload.sections)} sections × '
        f'{mode_info["queries"]} queries × {len(mode_info["tiers"])} tier(s) = '
        f'<strong>{mode_info["api_calls"]}</strong> estimated API calls '
        f'(est. cost: <strong>{mode_info["cost"]}</strong>)'
        f"</div>",
        unsafe_allow_html=True,
    )

    if st.button("▶ Start Experiment", type="primary", use_container_width=True):
        experiment_config = ExperimentConfig(
            mode=ExperimentMode(selected_mode),
            quality_tolerance=quality_tolerance,
            redundancy_threshold=redundancy_threshold,
            repetitions=mode_info["reps"],
            reasoning_tiers=mode_info["tiers"],
        )
        st.session_state["experiment_config"] = experiment_config

        # Clear stale results from a previous run
        for key in ("ablation_results", "diet_plan", "report_html",
                    "experiment_running", "experiment_queue",
                    "experiment_phase", "experiment_progress"):
            st.session_state.pop(key, None)

        st.switch_page("pages/2_progress.py")
