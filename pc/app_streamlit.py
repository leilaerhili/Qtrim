import base64
from io import BytesIO
from pathlib import Path
from typing import Optional

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from qiskit import QuantumCircuit

API_URL = "http://127.0.0.1:8000/optimize"

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"

CIRCUITS = {
    "Parity": "parity",
    "Half Adder": "half_adder",
    "Majority Vote": "majority_vote",
    "Linear Dataflow Pipeline": "linear_dataflow_pipeline",
}

BASELINE = {
    "parity": {"gate_count": 64, "depth": 24, "cost": 118},
    "half_adder": {"gate_count": 72, "depth": 28, "cost": 132},
    "majority_vote": {"gate_count": 80, "depth": 31, "cost": 150},
    "linear_dataflow_pipeline": {"gate_count": 95, "depth": 40, "cost": 190},
}

DEFAULT_CIRCUIT_STYLE = {
    "backgroundcolor": "none",
    "gatefacecolor": "none",
    "barrierfacecolor": "none",
    "linecolor": "#e8f1ff",
    "textcolor": "#e8f1ff",
    "gatetextcolor": "#e8f1ff",
    "subtextcolor": "#b9e3ff",
    "barrieredgecolor": "#b9e3ff",
}

st.set_page_config(page_title="QTrim", layout="wide")

# ---------- Theme helpers ----------
def b64_file(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()

def inject_theme(bg_path: Path):
    bg = b64_file(bg_path)
    st.markdown(
        f"""
        <style>
          .stApp {{
            background: url("data:image/png;base64,{bg}") no-repeat center center fixed;
            background-size: cover;
          }}

          .block-container {{
            padding-top: 1.2rem;
            padding-bottom: 2rem;
          }}

          #MainMenu {{visibility: hidden;}}
          footer {{visibility: hidden;}}
          header {{visibility: hidden;}}

          .panel {{
            background: rgba(10, 14, 22, 0.55);
            border: 1px solid rgba(120, 170, 255, 0.18);
            border-radius: 16px;
            padding: 16px;
            backdrop-filter: blur(10px);
            box-shadow: 0 0 0 1px rgba(255,255,255,0.04) inset;
          }}

          .section-title {{
            font-size: 1.05rem;
            letter-spacing: 0.4px;
            opacity: 0.92;
            margin-bottom: 10px;
          }}

          .kpi {{
            background: rgba(8, 12, 18, 0.55);
            border: 1px solid rgba(120, 170, 255, 0.16);
            border-radius: 14px;
            padding: 12px 14px;
          }}
          .kpi-label {{opacity: 0.72; font-size: 0.85rem;}}
          .kpi-value {{font-size: 1.9rem; font-weight: 700; line-height: 1.1;}}
          .kpi-delta {{opacity: 0.85; margin-top: 4px;}}

          div.stButton > button {{
            border-radius: 12px !important;
            border: 1px solid rgba(120, 170, 255, 0.30) !important;
            background: linear-gradient(180deg, rgba(90,160,255,0.28), rgba(20,40,80,0.35)) !important;
            color: rgba(235, 245, 255, 0.95) !important;
            padding: 0.6rem 1rem !important;
            font-weight: 600 !important;
          }}

          button[data-baseweb="tab"] {{
            font-size: 0.95rem;
            padding-top: 10px;
            padding-bottom: 10px;
          }}

        </style>
        """,
        unsafe_allow_html=True,
    )

def centered_logo_png(path: Path, width_px: int = 260, nudge_px: int = 0):
    """
    Streamlit's st.image() can look off-center if the PNG has uneven transparent padding.
    This renders the PNG as inline HTML and allows a small translateX nudge.
    """
    logo_b64 = b64_file(path)
    st.markdown(
        f"""
        <div style="display:flex; justify-content:center; width:100%;">
          <img src="data:image/png;base64,{logo_b64}"
               style="
                 width:{width_px}px;
                 height:auto;
                 transform: translateX({nudge_px}px);
                 filter: drop-shadow(0 0 10px rgba(120,170,255,0.25));
               " />
        </div>
        """,
        unsafe_allow_html=True,
    )

def pct_change(before: int, after: int) -> float:
    if before == 0:
        return 0.0
    return (before - after) / before * 100.0

def kpi(label: str, value: int, delta_pct: float):
    arrow = "↓" if delta_pct >= 0 else "↑"
    st.markdown(
        f"""
        <div class="kpi">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value">{value}</div>
          <div class="kpi-delta">{arrow} {abs(delta_pct):.1f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def bar_chart(before: dict, after: dict):
    metrics = ["Circuit Cost", "Gate Count", "Depth"]
    before_vals = [before["cost"], before["gate_count"], before["depth"]]
    after_vals  = [after["cost"],  after["gate_count"],  after["depth"]]

    fig = go.Figure()
    fig.add_bar(name="Before", x=metrics, y=before_vals)
    fig.add_bar(name="After", x=metrics, y=after_vals)
    fig.update_layout(
        barmode="group",
        height=340,
        margin=dict(l=10, r=10, t=25, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="rgba(235,245,255,0.92)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

def load_qasm(qasm_text: str) -> Optional[QuantumCircuit]:
    if not qasm_text or not qasm_text.strip():
        return None
    try:
        return QuantumCircuit.from_qasm_str(qasm_text)
    except Exception:
        try:
            from qiskit import qasm2

            return qasm2.loads(qasm_text)
        except Exception:
            return None

def circuit_to_png_bytes(circuit: QuantumCircuit) -> Optional[bytes]:
    try:
        import matplotlib.pyplot as plt

        fig = circuit.draw(output="mpl", style=DEFAULT_CIRCUIT_STYLE)
        fig.patch.set_alpha(0)
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=200, bbox_inches="tight", transparent=True)
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()
    except Exception:
        return None

def format_ascii_diagram(diagram: str) -> str:
    lines = diagram.splitlines()
    wire_start = None
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("q_") or stripped.startswith("c_"):
            colon = line.find(":")
            if colon != -1:
                wire_start = colon + 2
                break
    if wire_start is None:
        return diagram

    aligned = []
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("q_") or stripped.startswith("c_"):
            aligned.append(line)
            continue
        lead = len(line) - len(line.lstrip(" "))
        pad = max(0, wire_start - lead)
        aligned.append((" " * pad) + line)
    return "\n".join(aligned)

def render_qasm_panel(title: str, qasm_text: Optional[str], empty_message: str) -> None:
    st.markdown(
        f"<div style='opacity:0.85; font-weight:600; margin-bottom:6px;'>{title}</div>",
        unsafe_allow_html=True,
    )
    if not qasm_text:
        st.info(empty_message)
        return

    circuit = load_qasm(qasm_text)
    if circuit is None:
        st.warning("Failed to parse QASM. Showing raw text instead.")
        st.code(qasm_text, language="text")
        return

    st.markdown(
        "<div style='opacity:0.7; font-size:0.85rem; margin-bottom:4px;'>Compact Diagram (Qiskit)</div>",
        unsafe_allow_html=True,
    )
    compact_png = circuit_to_png_bytes(circuit)
    if compact_png is None:
        st.warning("Failed to render compact Qiskit diagram.")
    else:
        st.image(compact_png, use_container_width=True)

    try:
        ascii_diagram = circuit.draw(output="text")
        ascii_diagram = format_ascii_diagram(str(ascii_diagram))
        st.markdown(
            "<div style='opacity:0.7; font-size:0.85rem; margin:10px 0 4px;'>Compact Diagram (ASCII)</div>",
            unsafe_allow_html=True,
        )
        st.code(ascii_diagram, language="text")
    except Exception:
        st.warning("Failed to render compact ASCII diagram.")
    with st.expander("Show QASM", expanded=False):
        st.code(qasm_text, language="text")

# ---------- Inject theme ----------
BG_PATH = ASSETS / "bg.png"
LOGO_PATH = ASSETS / "QTrim_Logo.png"

inject_theme(BG_PATH)

# ---------- Header (FULL-WIDTH centered, visually corrected logo) ----------
# Adjust this if the logo still looks slightly off-center due to PNG padding.
LOGO_NUDGE_PX = 18     # try 0, 10, 18, 24 until perfect on your screen
LOGO_WIDTH_PX = 280    # make it bigger here

centered_logo_png(LOGO_PATH, width_px=LOGO_WIDTH_PX, nudge_px=LOGO_NUDGE_PX)

st.markdown(
    """
    <div style="
        text-align:center;
        margin-top:8px;
        font-size:1.9rem;
        font-weight:700;
        color: rgba(235,245,255,0.95);
    ">QTrim</div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style="
        text-align:center;
        opacity:0.75;
        margin-top:-6px;
        font-size:1rem;
    ">Reinforcement Learning for Quantum Circuit Optimization</div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)

# ---------- Session state ----------
if "selected_circuit_id" not in st.session_state:
    st.session_state.selected_circuit_id = "half_adder"
if "before" not in st.session_state:
    st.session_state.before = BASELINE["half_adder"].copy()
if "after" not in st.session_state:
    st.session_state.after = None
if "before_qasm" not in st.session_state:
    st.session_state.before_qasm = None
if "after_qasm" not in st.session_state:
    st.session_state.after_qasm = None
if "resolved_profile_id" not in st.session_state:
    st.session_state.resolved_profile_id = None

# ---------- Main layout ----------
left, right = st.columns([0.28, 0.72], gap="large")

# LEFT PANEL: only circuit selection + run
with left:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Quantum Circuit Optimizer</div>", unsafe_allow_html=True)

    circuit_label = st.selectbox("Circuit Type", list(CIRCUITS.keys()))
    circuit_id = CIRCUITS[circuit_label]
    profile_id = "auto"
    st.caption("Priority profile is selected on the phone.")
    if st.session_state.resolved_profile_id:
        st.markdown(
            f"Phone Priority: `{st.session_state.resolved_profile_id}`"
        )

    with st.expander("Run Constraints", expanded=False):
        max_depth_budget = st.number_input("Max Depth", min_value=0, value=0, step=1)
        max_latency_ms = st.number_input("Max Latency (ms)", min_value=0, value=0, step=50)
        max_shots = st.number_input("Max Shots", min_value=0, value=0, step=100)
        queue_level = st.selectbox("Queue Level", ["low", "normal", "high"], index=1)
        noise_level = st.selectbox("Noise Level", ["low", "normal", "high"], index=1)
        backend_condition = st.text_input("Backend", value="unknown")

    # Update baseline if selection changes
    if circuit_id != st.session_state.selected_circuit_id:
        st.session_state.selected_circuit_id = circuit_id
        st.session_state.before = BASELINE[circuit_id].copy()
        st.session_state.after = None
        st.session_state.before_qasm = None
        st.session_state.after_qasm = None
        st.session_state.resolved_profile_id = None

    run = st.button("Run QTrim", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if run:
        budgets = {
            "max_depth": int(max_depth_budget) if int(max_depth_budget) > 0 else None,
            "max_latency_ms": float(max_latency_ms) if float(max_latency_ms) > 0 else None,
            "max_shots": int(max_shots) if int(max_shots) > 0 else None,
        }
        payload = {
            "circuit_id": st.session_state.selected_circuit_id,
            "profile_id": profile_id,
            "budgets": budgets,
            "context": {
                "queue_level": queue_level,
                "noise_level": noise_level,
                "backend": backend_condition,
            },
        }
        with st.spinner("Optimizing..."):
            try:
                resp = requests.post(API_URL, json=payload, timeout=12)
                resp.raise_for_status()
                result = resp.json()

                st.session_state.before = result["before"]
                st.session_state.after = result["after"]
                st.session_state.resolved_profile_id = result.get("profile_id")

                # Future: RL can return these
                st.session_state.before_qasm = result.get("before_qasm")
                st.session_state.after_qasm = result.get("after_qasm")
            except Exception as e:
                st.error(f"API call failed: {e}")

# RIGHT PANEL: tabs (Circuit View / Optimization Metrics)
with right:
    tabs = st.tabs(["Circuit View", "Optimization Metrics"])

    # --- Circuit View tab ---
    with tabs[0]:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Circuit View</div>", unsafe_allow_html=True)

        bcol, acol = st.columns(2, gap="large")

        with bcol:
            render_qasm_panel(
                "Untrimmed Circuit",
                st.session_state.before_qasm,
                "Circuit diagram placeholder (before)",
            )

        with acol:
            render_qasm_panel(
                "Trimmed Circuit",
                st.session_state.after_qasm,
                "Run QTrim to generate optimized circuit",
            )

        st.markdown("</div>", unsafe_allow_html=True)

    # --- Optimization Metrics tab ---
    with tabs[1]:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Optimization Metrics</div>", unsafe_allow_html=True)

        before = st.session_state.before
        after = st.session_state.after

        k1, k2, k3 = st.columns(3, gap="medium")

        if after:
            d_cost = pct_change(before["cost"], after["cost"])
            d_gate = pct_change(before["gate_count"], after["gate_count"])
            d_depth = pct_change(before["depth"], after["depth"])

            with k1: kpi("Circuit Cost", after["cost"], d_cost)
            with k2: kpi("Gate Count", after["gate_count"], d_gate)
            with k3: kpi("Depth", after["depth"], d_depth)

            df = pd.DataFrame(
                [
                    ["Gate Count", before["gate_count"], after["gate_count"], f"{d_gate:.1f}%"],
                    ["Depth", before["depth"], after["depth"], f"{d_depth:.1f}%"],
                    ["Circuit Cost", before["cost"], after["cost"], f"{d_cost:.1f}%"],
                ],
                columns=["Metric", "Before", "After", "Δ %"],
            )
            st.dataframe(df, use_container_width=True, hide_index=True)
            bar_chart(before, after)

            st.success(f"Optimization reduced circuit cost by {d_cost:.1f}%.")
        else:
            with k1: kpi("Circuit Cost", before["cost"], 0.0)
            with k2: kpi("Gate Count", before["gate_count"], 0.0)
            with k3: kpi("Depth", before["depth"], 0.0)
            st.info("Run QTrim to populate metrics.")

        st.markdown("</div>", unsafe_allow_html=True)
