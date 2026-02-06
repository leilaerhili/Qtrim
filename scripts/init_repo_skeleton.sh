#!/usr/bin/env bash
# scripts/init_repo_skeleton.sh
# Creates the quantum-multiverse-optimizer repo skeleton.

set -euo pipefail

ROOT_DIR="."

mkdir -p "$ROOT_DIR"/{pc,core,phone/android,core/policy_store,scripts}

# Top-level files
cat > "$ROOT_DIR/README.md" <<'EOF'
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
EOF

cat > "$ROOT_DIR/LICENSE" <<'EOF'
MIT License

Copyright (c) 2026

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
EOF

cat > "$ROOT_DIR/requirements.txt" <<'EOF'
# Core
qiskit>=1.0.0
gymnasium>=0.29.0

# RL
stable-baselines3>=2.2.1
torch>=2.1.0

# PC control surface + API
fastapi>=0.110.0
uvicorn[standard]>=0.27.0
streamlit>=1.31.0

# Optional utilities
numpy>=1.26.0
EOF

# Core placeholders
cat > "$ROOT_DIR/core/circuits_baseline.py" <<'EOF'
"""
Predefined baseline circuits + padding utilities.
"""
EOF

cat > "$ROOT_DIR/core/rewrites.py" <<'EOF'
"""
Rewrite rules (actions) for circuit optimization.
"""
EOF

cat > "$ROOT_DIR/core/metrics.py" <<'EOF'
"""
Metrics: gate count, depth, cost, (optional) correctness proxy.
"""
EOF

cat > "$ROOT_DIR/core/env_quantum_opt.py" <<'EOF'
"""
Gymnasium environment wrapping circuit optimization as RL.
"""
EOF

cat > "$ROOT_DIR/core/train_policy.py" <<'EOF'
"""
Train PPO/DQN policy and save to core/policy_store/.
"""
EOF

cat > "$ROOT_DIR/core/shared_schema.py" <<'EOF'
"""
Shared JSON schema for observations/actions between PC and phone.
"""
EOF

# PC placeholders
cat > "$ROOT_DIR/pc/app_streamlit.py" <<'EOF'
import streamlit as st

st.title("Quantum Multiverse Optimizer (placeholder)")
st.write("Control surface UI will live here.")
EOF

cat > "$ROOT_DIR/pc/api_server.py" <<'EOF'
from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
def health():
    return {"ok": True}
EOF

cat > "$ROOT_DIR/pc/demo_run.ps1" <<'EOF'
# One-command demo runner (placeholder).
# Start API server:
#   uvicorn pc.api_server:app --host 0.0.0.0 --port 8000
# Start UI:
#   streamlit run pc/app_streamlit.py
Write-Host "Run:"
Write-Host "  uvicorn pc.api_server:app --host 0.0.0.0 --port 8000"
Write-Host "  streamlit run pc/app_streamlit.py"
EOF

# Phone placeholders
cat > "$ROOT_DIR/phone/client.py" <<'EOF'
"""
Phone-side client placeholder (Termux/Python).
"""
EOF

cat > "$ROOT_DIR/phone/android/README.md" <<'EOF'
# Android Client Placeholder

This folder is reserved for an Android (Kotlin) client if we choose that path.
EOF

# Helper script to print tree
cat > "$ROOT_DIR/scripts/print_tree.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
if command -v tree >/dev/null 2>&1; then
  tree -a
else
  echo "Install 'tree' or use: find . -maxdepth 3 -print"
  find . -maxdepth 3 -print
fi
EOF
chmod +x "$ROOT_DIR/scripts/print_tree.sh"

echo "Created repo skeleton at: $ROOT_DIR"
echo "Next:"
echo "  cd $ROOT_DIR"
echo "  ./scripts/print_tree.sh"
