# Quantum Multiverse Optimizer

Multi-device RL-based quantum circuit optimizer demo for the Snapdragon Multiverse Hackathon.

## Quick start
1) Create a venv and install deps:
   - python -m venv .venv
   - source .venv/bin/activate  (Windows: .venv\Scripts\activate)
   - pip install -r requirements.txt

2) Run the control surface (placeholder):
   - streamlit run pc/app_streamlit.py

3) Run the PC API server (placeholder):
   - uvicorn pc.api_server:app --host 0.0.0.0 --port 8000

## Repo layout
- core/: circuits, rewrites, metrics, RL environment, training
- pc/: Streamlit control surface + FastAPI server + demo scripts
- phone/: Android or Termux client
