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

## Packaged EXE + APK flow (Windows + Android)
Run from repo root (`C:\Users\qc_de\Desktop\Qtrim`).

1) Build packaged artifacts:
   - `powershell -ExecutionPolicy Bypass -File scripts\build_exe.ps1`
   - `powershell -ExecutionPolicy Bypass -File scripts\build_apk.ps1`
   - Outputs:
     - `dist\windows\QTrimDesktop.exe`
     - `dist\android\QTrim-debug.apk`
     - `dist\android\QTrim-release-unsigned.apk` (release build, unsigned)

2) Install APK on phone:
   - If `adb` is on PATH:
     - `adb install -r dist\android\QTrim-debug.apk`
   - If not on PATH:
     - `& "C:\Users\qc_de\AppData\Local\Android\Sdk\platform-tools\adb.exe" install -r dist\android\QTrim-debug.apk`

3) Launch phone app:
   - Open **QTrim Phone** from app drawer, or:
   - `& "C:\Users\qc_de\AppData\Local\Android\Sdk\platform-tools\adb.exe" shell am start -n com.qtrim.android/.MainActivity`

4) USB inference tunnel (no phone IP needed):
   - `& "C:\Users\qc_de\AppData\Local\Android\Sdk\platform-tools\adb.exe" forward tcp:9002 tcp:9002`
   - Set env var in the same PowerShell where you run the EXE:
     - `$env:QTRIM_ANDROID_INFER_URL = "http://127.0.0.1:9002/infer"`

5) Start desktop packaged app:
   - `dist\windows\QTrimDesktop.exe`
   - Streamlit UI: `http://127.0.0.1:8501`
   - API: `http://127.0.0.1:8000`

6) Use flow:
   - Keep phone app open in foreground.
   - Choose profile on phone.
   - Click **Run QTrim** in desktop UI.

### Troubleshooting packaged flow
- `adb` not recognized:
  - Use full path: `C:\Users\qc_de\AppData\Local\Android\Sdk\platform-tools\adb.exe`
- App installed but not opened:
  - Install only copies APK; launch manually or with `adb shell am start ...`
- Desktop UI returns `502`:
  - Verify forwarding: `adb forward --list`
  - Verify phone endpoint:
    - `Invoke-WebRequest http://127.0.0.1:9002/health`
    - `Invoke-WebRequest http://127.0.0.1:9002/profile`
  - If needed, recreate tunnel:
    - `adb forward --remove-all`
    - `adb forward tcp:9002 tcp:9002`

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
  - `python -m core.train_policy --algo dqn --device directml --strict-device --baseline parity --timesteps 50000 --seeds 0,1,2 --n-steps 64 --batch-size 64 --save-name dqn_parity_50k`
- Mixed-baseline training + parallel env rollout + periodic training curves:
  - `python -m core.train_policy --algo dqn --device directml --strict-device --train-mode mixed --n-envs 1 --n-steps 64 --batch-size 64 --timesteps 50000 --eval-every 5000 --curve-eval-episodes 3 --save-name dqn_mixed_50k`
- Curriculum training (easy -> medium -> hard) with unseen holdout validation:
  - `python -m core.train_curriculum --timesteps 60000 --seeds 0,1,2 --ent-coef 0.01 --n-steps 1024 --save-name ppo_curriculum_60k`
- Both trainers auto-select and save the best holdout seed policy:
  - `core/policy_store/<save-name>_best.zip`
- Priority-profile aware runs (phone-style objectives + budgets):
  - `python -m core.train_policy --algo dqn --priority-profile-id high_fidelity --max-depth-budget 250 --max-latency-ms 1500 --max-shots 2000 --queue-level high --noise-level high --timesteps 50000 --save-name dqn_priority_demo`
  - Optional weight override:
  - `--priority-weights-json "{\"two_qubit_gates\":0.50,\"depth\":0.30,\"total_gates\":0.10,\"swap_gates\":0.10}"`

## Phone Priority Payload
- API now accepts either preset profile mode or explicit weights:
  - `profile_id`: `high_fidelity`, `low_latency`, `low_cost`, `balanced`
  - `weights`: optional metric weights override
  - `budgets`: optional hard limits (`max_depth`, `max_latency_ms`, `max_shots`)
  - `context`: optional backend conditions (`queue_level`, `noise_level`, `backend`)

## Android ONNX Export
- `core.train_policy` now exports Android-friendly ONNX artifacts by default per seed run:
  - `<save-name>_seedX_android_fp32_bs1.onnx` (fixed-shape FP32)
  - `<save-name>_seedX_android_int8_bs1.onnx` (optional dynamic-quantized INT8)
- Strict QNN-only validation (CPU fallback disabled) is also run by default and written into summary JSON under each run:
  - `runs[].android_onnx.qnn_validation_fp32`
  - `runs[].android_onnx.qnn_validation_int8`
- Useful flags:
  - `--no-export-android-onnx`
  - `--android-onnx-static-batch 1`
  - `--no-android-onnx-int8`
  - `--no-android-qnn-strict-check`

## Accelerator-first runs
- Device selection now supports: `auto`, `npu`, `directml`, `cuda`, `xpu`, `mps`, `cpu`.
- Default `core.train_policy` is now `dqn + directml`.
- DQN now supports ONNX Runtime rollout inference backends:
  - `--inference-backend ort-qnn` (prefer NPU via QNNExecutionProvider)
  - `--inference-backend ort-auto` (qnn > directml > cuda > cpu)
  - `--inference-backend torch` (disable ONNX runtime path)
- Force DirectML and fail if unavailable:
  - `python -m core.train_policy --algo dqn --device directml --strict-device --train-mode mixed --n-envs 1 --n-steps 64 --batch-size 64 --timesteps 50000 --save-name dqn_directml`
- Force QNN and fail if unavailable:
  - `python -m core.train_policy --algo dqn --inference-backend ort-qnn --strict-inference-backend --device directml --train-mode mixed --timesteps 50000 --save-name dqn_qnn`
- If `torch.npu` is unavailable, `--device npu` falls back to DirectML when `torch-directml` is installed (unless `--strict-device` is set).
- Training summaries now include `device_resolution`, `inference_resolution` (ONNX provider info), and `nexa_probe` per run.

Note: the RL policy network can run on accelerator backends, but circuit rewrites/Qiskit environment steps are still CPU-side.
