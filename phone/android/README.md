# Android Inference Handoff

Use `infer_server.py` to run ONNX policy inference on Android (Termux/Python) and
serve action predictions to the PC API.

## Recommended: Native Android app (no Termux)

If you do not want to run Termux, use the native app instead:

```
Qtrim/phone/android_app
```

Open it in Android Studio and follow the README there.

## Start server on Android

```bash
python phone/android/infer_server.py \
  --model core/policy_store/tiny_infer_handoff_seed0_android_int8_bs1.onnx \
  --host 0.0.0.0 \
  --port 9001 \
  --profile balanced
```

## Pick the priority profile on Android

Open the phone browser to the local UI:

```
http://127.0.0.1:9001/
```

That page lets you choose `balanced`, `high_fidelity`, `low_latency`, or `low_cost`.
The PC API will fetch the current selection from `GET /profile` when you send
`profile_id: "auto"` from the website.

## PC API configuration

Set the Android inference URL before starting FastAPI on PC:

```bash
export QTRIM_ANDROID_INFER_URL=http://<android-ip>:9001/infer
uvicorn pc.api_server:app --host 0.0.0.0 --port 8000
```

If you use USB port forwarding, point to `http://127.0.0.1:<forwarded-port>/infer`.

## Request/Response contract

- Request fields used by `/infer`:
  - `observation_vector`: list of 6 floats (preferred)
  - `observation`: schema payload fallback (optional)
  - `action_mask`: list of 0/1 allowed actions (optional)
- Response:
  - `{"action_id": <int>}`
