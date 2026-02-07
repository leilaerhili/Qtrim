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

## Baseline env demo
Run the RL environment demo with a selectable baseline circuit:
- python -m core.env_quantum_opt --baseline toy
- python -m core.env_quantum_opt --baseline parity
- python -m core.env_quantum_opt --baseline line --pad-level 3 --max-steps 15

Sample metrics (single run, random actions):
- toy: reset cost 8.0 (gate_count=6, depth=4, cx=2) -> best cost 6.5
- parity: reset cost varies by pad-level and random action sequence
- line: reset cost 10.5 (gate_count=7, depth=7, cx=3) -> best cost 9.0
Note: results vary by random action sequence.

## Repo layout
- core/: circuits, rewrites, metrics, RL environment, training
- pc/: Streamlit control surface + FastAPI server + demo scripts
- phone/: Android or Termux client

## Training
- Single baseline, multi-seed with holdout evaluation:
  - `python -m core.train_policy --baseline parity --timesteps 50000 --seeds 0,1,2 --ent-coef 0.01 --n-steps 1024 --save-name ppo_parity_50k`
- Mixed-baseline training + parallel env rollout + periodic training curves:
  - `python -m core.train_policy --train-mode mixed --n-envs 4 --timesteps 50000 --eval-every 5000 --curve-eval-episodes 3 --save-name ppo_mixed_50k`
- Curriculum training (easy -> medium -> hard) with unseen holdout validation:
  - `python -m core.train_curriculum --timesteps 60000 --seeds 0,1,2 --ent-coef 0.01 --n-steps 1024 --save-name ppo_curriculum_60k`
- Both trainers auto-select and save the best holdout seed policy:
  - `core/policy_store/<save-name>_best.zip`

## NPU-first runs
- Device selection now supports: `auto`, `npu`, `directml`, `cuda`, `xpu`, `mps`, `cpu`.
- Force NPU and fail if it is not available:
  - `python -m core.train_policy --device npu --strict-device --train-mode mixed --n-envs 4 --timesteps 50000 --save-name ppo_npu`
  - `python -m core.train_curriculum --device npu --strict-device --timesteps 60000 --save-name ppo_curriculum_npu`
- If `torch.npu` is unavailable, `--device npu` falls back to DirectML when `torch-directml` is installed (unless `--strict-device` is set).
- Training summaries now include `device_resolution` per run so you can show exactly which backend/device was used.

Note: the RL policy network can run on accelerator backends, but circuit rewrites/Qiskit environment steps are still CPU-side.
