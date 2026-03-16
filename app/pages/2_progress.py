"""
app/pages/2_progress.py — Experiment progress and live monitoring.

Launches the ablation sweep in a background daemon thread and polls the
progress queue every 2 seconds via @st.fragment. Never calls st.* from
the worker thread — all communication is via queue.Queue.

Session state written:
  - bedrock_client    (BedrockClient)
  - experiment_queue  (queue.Queue)
  - experiment_running (bool)
  - experiment_phase  (str)
  - experiment_progress (dict)
  - ablation_results  (AblationResults)
"""

from __future__ import annotations

import time
from queue import Empty, Queue
from threading import Thread

import streamlit as st

from app.components.layout import apply_layout
from core.ablation_engine import run_full_sweep
from infra.bedrock_client import BedrockClient

apply_layout()

# ── Worker (background thread — NO st.* calls here) ──────────────────────────


def _ablation_worker(
    client: BedrockClient,
    payload,
    config,
    result_queue: Queue,
) -> None:
    """Execute the full ablation sweep and post results to the queue.

    This function runs in a daemon thread. It must never call any st.*
    function — Streamlit is not thread-safe for cross-thread calls.
    All output goes through result_queue.put().
    """
    try:
        results = run_full_sweep(
            client=client,
            payload=payload,
            config=config,
            progress_queue=result_queue,
        )
        result_queue.put({"type": "final_results", "data": results})
    except Exception as exc:  # noqa: BLE001
        result_queue.put({"type": "fatal", "error": str(exc)})


# ── Session state initialisation ──────────────────────────────────────────────


def _init_state() -> None:
    """Set default session state values on first page load."""
    defaults: dict = {
        "experiment_queue":    None,
        "experiment_running":  False,
        "experiment_phase":    "",
        "experiment_progress": {"completed": 0, "total": 0, "errors": 0},
        "ablation_results":    None,
        "bedrock_client":      None,
        "_exp_start_time":     None,
        "_exp_errors":         [],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_state()

# ── Guards ────────────────────────────────────────────────────────────────────

payload = st.session_state.get("payload")
config  = st.session_state.get("experiment_config")

st.title("⚙️ Experiment Progress")

if payload is None or config is None:
    st.warning(
        "No experiment configured. Please upload a context payload and select "
        "an experiment mode first."
    )
    if st.button("← Back to Upload"):
        st.switch_page("pages/1_upload.py")
    st.stop()

# ── Launch experiment on first visit (or if not already running) ──────────────

if not st.session_state["experiment_running"] and st.session_state["ablation_results"] is None:
    st.markdown(
        f'<div style="background:#f0ebff; border:1px solid #c4b5fd; border-radius:10px; '
        f'padding:0.9rem 1.1rem; margin-bottom:1.2rem;">'
        f'<span style="font-weight:600; color:#7c6fea;">Starting experiment:</span> '
        f'<strong>{config.mode.value.capitalize()} mode</strong> — '
        f'{len(payload.sections)} sections, '
        f'{len(config.reasoning_tiers)} reasoning tier(s), '
        f'{config.repetitions} repetition(s)'
        f'</div>',
        unsafe_allow_html=True,
    )

    with st.spinner("Initialising Bedrock client…"):
        try:
            client = BedrockClient()
            st.session_state["bedrock_client"] = client
        except Exception as exc:  # noqa: BLE001
            st.error(
                f"Failed to initialise Bedrock client: {exc}\n\n"
                "Check that your AWS credentials are set in `.env` "
                "(see `.env.example` for the required keys)."
            )
            st.stop()

    # Fresh queue and state for this run
    result_queue = Queue()
    st.session_state["experiment_queue"]    = result_queue
    st.session_state["experiment_running"]  = True
    st.session_state["experiment_phase"]    = "Starting…"
    st.session_state["experiment_progress"] = {"completed": 0, "total": 0, "errors": 0}
    st.session_state["_exp_start_time"]     = time.monotonic()
    st.session_state["_exp_errors"]         = []

    Thread(
        target=_ablation_worker,
        args=(client, payload, config, result_queue),
        daemon=True,
    ).start()

# ── Progress display (polls queue every 2 seconds) ────────────────────────────

@st.fragment(run_every="2s")
def _progress_display() -> None:
    """Drain all available queue messages and update UI. Runs every 2 s."""
    q: Queue | None = st.session_state.get("experiment_queue")
    if q is None:
        return

    # Drain all pending messages without blocking
    while True:
        try:
            msg = q.get_nowait()
            _handle_message(msg)
        except Empty:
            break

    _render_progress_ui()


def _handle_message(msg: dict) -> None:
    """Update session state based on a single queue message."""
    msg_type = msg.get("type", "")
    prog     = st.session_state["experiment_progress"]

    if msg_type == "start":
        prog["total"] = msg.get("total", 0)
        st.session_state["experiment_phase"] = "Running baseline evaluation…"

    elif msg_type in ("baseline_rep_complete", "baseline_complete"):
        rep       = msg.get("rep", "")
        total_rep = msg.get("total_reps", "")
        completed = msg.get("completed", prog["completed"])
        total     = msg.get("total", prog["total"])
        prog["completed"] = completed
        prog["total"]     = max(prog["total"], total)
        label = (
            f"Baseline evaluation ({rep}/{total_rep} reps)…"
            if rep else "Baseline evaluation complete"
        )
        st.session_state["experiment_phase"] = label

    elif msg_type == "section_complete":
        prog["completed"] = msg.get("completed", prog["completed"] + 1)
        prog["total"]     = max(prog["total"], msg.get("total", prog["total"]))
        section_id        = msg.get("section_id", "")
        st.session_state["experiment_phase"] = (
            f"Ablating sections… ({prog['completed']}/{prog['total']})"
            + (f" — {section_id}" if section_id else "")
        )

    elif msg_type == "sweep_complete":
        st.session_state["experiment_phase"] = "Single-section sweep complete. Running analysis…"

    elif msg_type == "elimination_start":
        n = msg.get("candidates", 0)
        st.session_state["experiment_phase"] = f"Multi-section elimination: {n} candidates…"

    elif msg_type == "elimination_complete":
        lean_reduction = msg.get("lean_reduction", 0.0)
        st.session_state["experiment_phase"] = (
            f"Lean configuration found — {lean_reduction:.1%} token reduction"
        )

    elif msg_type == "ordering_start":
        n = msg.get("candidates", 0)
        st.session_state["experiment_phase"] = f"Ordering experiments: {n} candidate sections…"

    elif msg_type == "ordering_progress":
        completed = msg.get("completed", 0)
        total     = msg.get("total", 0)
        st.session_state["experiment_phase"] = f"Ordering experiments ({completed}/{total})…"

    elif msg_type == "ordering_complete":
        st.session_state["experiment_phase"] = "Ordering experiments complete."

    elif msg_type == "error":
        prog["errors"] += 1
        st.session_state["_exp_errors"].append(
            f"Section `{msg.get('section_id', '?')}`: {msg.get('message', str(msg))}"
        )

    elif msg_type == "done":
        st.session_state["experiment_phase"] = "Post-processing results…"

    elif msg_type == "final_results":
        st.session_state["ablation_results"]  = msg["data"]
        st.session_state["experiment_running"] = False
        st.session_state["experiment_phase"]   = "Complete ✅"

    elif msg_type == "fatal":
        st.session_state["experiment_running"] = False
        st.session_state["experiment_phase"]   = f"Failed ❌: {msg.get('error', 'unknown error')}"
        st.session_state["_exp_errors"].append(f"Fatal: {msg.get('error', '')}")


def _render_progress_ui() -> None:
    """Render the current progress state as Streamlit UI elements."""
    prog      = st.session_state["experiment_progress"]
    phase     = st.session_state["experiment_phase"]
    running   = st.session_state["experiment_running"]
    results   = st.session_state.get("ablation_results")
    errors    = st.session_state["_exp_errors"]
    start_t   = st.session_state.get("_exp_start_time")

    completed = prog["completed"]
    total     = max(prog["total"], 1)
    n_errors  = prog["errors"]
    elapsed   = round(time.monotonic() - start_t) if start_t else 0

    # Phase label
    st.markdown(
        f'<div style="font-size:0.95rem; font-weight:600; color:#374151; '
        f'margin-bottom:0.75rem;">{phase}</div>',
        unsafe_allow_html=True,
    )

    # Progress bar
    if running or completed > 0:
        bar_val = min(1.0, completed / total) if total > 0 else 0.0
        st.progress(
            bar_val,
            text=f"{completed}/{total} experiments" if total > 1 else "Running…",
        )

    # Metrics row
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Completed", f"{completed}/{total}")
    with col2:
        st.metric("Errors", n_errors)
    with col3:
        mins, secs = divmod(elapsed, 60)
        st.metric("Elapsed", f"{mins}m {secs:02d}s")

    # Fatal error display
    if not running and "Fatal" in phase:
        st.error(f"Experiment failed. See error log below.")

    # Error log expander
    if errors:
        with st.expander(f"⚠️ Errors ({len(errors)})", expanded=False):
            for e in errors:
                st.markdown(
                    f'<div style="font-size:0.82rem; color:#dc2626; '
                    f'padding:4px 8px; background:#fef2f2; border-radius:6px; '
                    f'margin-bottom:4px;">{e}</div>',
                    unsafe_allow_html=True,
                )

    # Completion state
    if results is not None and not running:
        n_sections = len(results.section_impacts)
        reduction  = results.lean_token_reduction * 100
        retention  = results.lean_quality_retention * 100
        cost       = results.total_cost

        st.success(
            f"🎉 Experiment complete! Analysed **{n_sections} sections** — "
            f"**{reduction:.1f}% token reduction** with **{retention:.1f}% quality retention** "
            f"(cost: **${cost:.4f}**)."
        )

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("📊 View Results →", type="primary", use_container_width=True):
                st.switch_page("pages/3_results.py")
        with col_b:
            if st.button("🥗 Context Diet Plan →", use_container_width=True):
                st.switch_page("pages/4_diet_plan.py")


_progress_display()
