# QTrim Android App (Native)

This app runs the ONNX policy on-device and serves the same HTTP API as the
Termux server. The PC API can call `http://<phone-ip>:9002/infer` directly.

## Open in Android Studio

1) Open `Qtrim/phone/android_app` in Android Studio.
2) Build and run on your device (minSdk 26).

## Use

- Keep the app open while using the PC UI.
- Select the priority profile in the app.
- The app shows the device IP and the `/infer` endpoint.

## PC configuration

```bash
export QTRIM_ANDROID_INFER_URL=http://<phone-ip>:9002/infer
uvicorn pc.api_server:app --host 0.0.0.0 --port 8000
```

## Endpoints

- `GET /health` -> `{"ok": true, "profile_id": "balanced"}`
- `GET /profile` -> `{"profile_id": "balanced", "allowed_profiles": [...]}`
- `POST /profile` -> `{"profile_id": "low_latency"}`
- `POST /infer` -> `{"action_id": 3}`

## Model

The model is bundled at:

```
app/src/main/assets/tiny_infer_handoff_seed0_android_int8_bs1.onnx
```

If you want to swap models, replace the file and rebuild.
