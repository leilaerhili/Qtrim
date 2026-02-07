"""
QTrim - Quantum Circuit Optimization UI
----------------------------------------

This version:

- Renders real Qiskit circuit diagrams
- Uses FAKE circuits for demo if backend doesn't provide QASM yet
- Is ready to accept real before_qasm / after_qasm from RL later
- Clean separation of responsibilities
"""

import base64
from pathlib import Path
from io import BytesIO
from typing import Optional, Union, Dict

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit


# -----------------------------
# Configuration
# -----------------------------

API_URL = "http://127.0.0.1:8000/optimize"

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"

CIRCUITS = {
    "Parity": "parity",
    "Half Adder": "half_adder",
    "Majority Vote": "majority_vote",
    "Linear Dataflow Pipeline": "linear_dataflow_pipeline",
}

# Temporary baseline metrics (until RL returns real ones)
BASELINE = {
    "parity": {"gate_count": 64, "depth": 24, "cost": 118},
    "half_adder": {"gate_count": 72, "depth": 28, "cost": 132},
    "majority_vote": {"gate_count": 80, "depth": 31, "cost": 150},
    "linear_dataflow_pipeline": {"gate_count": 95, "depth": 40, "cost": 190},
}

DEFAULT_CIRCUIT_STYLE: Dict[str, str] = {
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


# -----------------------------
# Utility: Render Qiskit Circuit
# -----------------------------

def set_background(image_path: Path) -> None:
    """
    Set a full-page background image for the Streamlit app.
    """
    try:
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    except Exception:
        return

    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(rgba(8, 12, 20, 0.75), rgba(8, 12, 20, 0.75)),
                        url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        .block-container {{
            padding-top: 0.5rem;
        }}
        .qtrim-logo {{
            display: block;
            margin: 4px auto 0 auto;
            width: 300px;
            max-width: 60vw;
            transform: translateX(-34px);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


set_background(ASSETS / "bg.png")

def image_to_base64(image_path: Path) -> Optional[str]:
    """
    Read an image from disk and return a base64 string.
    """
    try:
        return base64.b64encode(image_path.read_bytes()).decode("ascii")
    except Exception:
        return None

def circuit_to_png_bytes(qc: QuantumCircuit, style: Optional[Dict[str, str]] = None) -> bytes:
    """
    Converts a QuantumCircuit to PNG bytes using matplotlib.
    """
    fig = qc.draw(output="mpl", style=style) if style else qc.draw(output="mpl")
    fig.patch.set_alpha(0)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight", transparent=True)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def render_circuit(
    title: str,
    qasm_or_circuit: Optional[Union[str, QuantumCircuit]],
    style: Optional[Dict[str, str]] = None,
):
    """
    Render circuit from a QASM string or an in-memory QuantumCircuit.
    """
    st.markdown(f"**{title}**")

    if qasm_or_circuit is None or (
        isinstance(qasm_or_circuit, str) and not qasm_or_circuit.strip()
    ):
        st.info("Circuit diagram will appear here.")
        return

    try:
        if isinstance(qasm_or_circuit, QuantumCircuit):
            qc = qasm_or_circuit
        else:
            qc = QuantumCircuit.from_qasm_str(qasm_or_circuit)
    except Exception as e:
        st.error(f"Failed to parse circuit: {e}")
        if isinstance(qasm_or_circuit, str):
            st.code(qasm_or_circuit)
        return

    try:
        img = circuit_to_png_bytes(qc, style=style)
        st.image(img, use_container_width=True)
        return
    except Exception as e:
        try:
            text_diagram = qc.draw(output="text")
            st.code(str(text_diagram))
            return
        except Exception:
            st.error(f"Failed to render circuit: {e}")
            if isinstance(qasm_or_circuit, str):
                st.code(qasm_or_circuit)


# -----------------------------
# Fake Circuits for Demo
# -----------------------------

def make_fake_circuits(circuit_id: str):
    """
    Generates fake before/after circuits for UI demo.
    Replace this later with real RL output.
    """

    if circuit_id == "half_adder":
        before = QuantumCircuit(3, 2)
        before.cx(0, 1)
        before.cx(1, 2)
        before.ccx(0, 1, 2)
        before.measure(1, 0)
        before.measure(2, 1)

        after = QuantumCircuit(3, 2)
        after.cx(0, 1)
        after.ccx(0, 1, 2)
        after.measure(1, 0)
        after.measure(2, 1)

    elif circuit_id == "parity":
        before = QuantumCircuit(4, 1)
        before.cx(0, 3)
        before.cx(1, 3)
        before.cx(2, 3)
        before.measure(3, 0)

        after = QuantumCircuit(4, 1)
        after.cx(0, 3)
        after.cx(1, 3)
        after.measure(3, 0)

    else:
        before = QuantumCircuit(5)
        before.h(0)
        before.cx(0, 1)
        before.rz(0.2, 1)
        before.rz(0.3, 1)
        before.cx(1, 2)

        after = QuantumCircuit(5)
        after.h(0)
        after.cx(0, 1)
        after.rz(0.5, 1)
        after.cx(1, 2)

    return before, after


# -----------------------------
# Metrics helpers
# -----------------------------

def pct_change(before, after):
    if before == 0:
        return 0
    return (before - after) / before * 100


def bar_chart(before, after):
    metrics = ["Cost", "Gate Count", "Depth"]
    fig = go.Figure()
    fig.add_bar(name="Before", x=metrics,
                y=[before["cost"], before["gate_count"], before["depth"]],
                width=0.25)
    fig.add_bar(name="After", x=metrics,
                y=[after["cost"], after["gate_count"], after["depth"]],
                width=0.25)
    fig.update_layout(barmode="group", height=350)
    st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# Header
# -----------------------------

logo_b64 = image_to_base64(ASSETS / "QTrim_Logo.png")
if logo_b64:
    st.markdown(
        f'<img class="qtrim-logo" src="data:image/png;base64,{logo_b64}" />',
        unsafe_allow_html=True,
    )
else:
    st.image(str(ASSETS / "QTrim_Logo.png"), width=240)

st.markdown("<h2 style='text-align:center; margin-top: -6px;'>QTrim</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Reinforcement Learning for Quantum Circuit Optimization</p>", unsafe_allow_html=True)


# -----------------------------
# Session State
# -----------------------------

if "selected" not in st.session_state:
    st.session_state.selected = "half_adder"
if "before" not in st.session_state:
    st.session_state.before = BASELINE["half_adder"]
if "after" not in st.session_state:
    st.session_state.after = None
if "before_qasm" not in st.session_state:
    st.session_state.before_qasm = None
if "after_qasm" not in st.session_state:
    st.session_state.after_qasm = None


# -----------------------------
# Layout
# -----------------------------

left, right = st.columns([0.3, 0.7])

# Left Panel
with left:
    st.markdown("### Circuit Selection")
    label = st.selectbox("Choose Algorithm", list(CIRCUITS.keys()))
    circuit_id = CIRCUITS[label]

    if circuit_id != st.session_state.selected:
        st.session_state.selected = circuit_id
        st.session_state.before = BASELINE[circuit_id]
        st.session_state.after = None

    if st.button("Run QTrim", use_container_width=True):

        payload = {
            "circuit_id": circuit_id,
            "constraint_profile": "low_noise"
        }

        try:
            resp = requests.post(API_URL, json=payload, timeout=8)
            result = resp.json()

            st.session_state.before = result["before"]
            st.session_state.after = result["after"]

            # If backend provides QASM later, use it
            if result.get("before_qasm") and result.get("after_qasm"):
                st.session_state.before_qasm = result["before_qasm"]
                st.session_state.after_qasm = result["after_qasm"]
            else:
                # Otherwise use fake demo circuits
                bq, aq = make_fake_circuits(circuit_id)
                st.session_state.before_qasm = bq
                st.session_state.after_qasm = aq

        except Exception:
            # If API fails, still generate fake circuits for demo
            bq, aq = make_fake_circuits(circuit_id)
            st.session_state.before_qasm = bq
            st.session_state.after_qasm = aq
            st.session_state.after = {
                "gate_count": int(st.session_state.before["gate_count"] * 0.8),
                "depth": int(st.session_state.before["depth"] * 0.8),
                "cost": int(st.session_state.before["cost"] * 0.75),
            }


# Right Panel
with right:

    tabs = st.tabs(["Circuit View", "Metrics"])

    # Circuit View
    with tabs[0]:
        col1, col2 = st.columns(2)
        with col1:
            render_circuit("Untrimmed Circuit", st.session_state.before_qasm, style=DEFAULT_CIRCUIT_STYLE)
        with col2:
            render_circuit("Trimmed Circuit", st.session_state.after_qasm, style=DEFAULT_CIRCUIT_STYLE)

    # Metrics View
    with tabs[1]:
        before = st.session_state.before
        after = st.session_state.after

        if after:
            df = pd.DataFrame([
                ["Gate Count", before["gate_count"], after["gate_count"], f"{pct_change(before['gate_count'], after['gate_count']):.1f}%"],
                ["Depth", before["depth"], after["depth"], f"{pct_change(before['depth'], after['depth']):.1f}%"],
                ["Cost", before["cost"], after["cost"], f"{pct_change(before['cost'], after['cost']):.1f}%"],
            ], columns=["Metric", "Before", "After", "Improvement"])

            st.dataframe(df, use_container_width=True)
            bar_chart(before, after)
        else:
            st.info("Run QTrim to generate results.")
