from __future__ import annotations

import argparse
import atexit
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
import webbrowser


def _frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def _resource_root() -> Path:
    if _frozen():
        return Path(getattr(sys, "_MEIPASS"))
    return Path(__file__).resolve().parents[1]


def _launcher_command() -> list[str]:
    if _frozen():
        return [sys.executable]
    return [sys.executable, str(Path(__file__).resolve())]


def _terminate_process(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=10)
    except Exception:
        proc.kill()


def _wait_for_http(url: str, timeout_s: float) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2):
                return True
        except Exception:
            time.sleep(0.5)
    return False


def _run_api_mode(host: str, port: int) -> int:
    import uvicorn

    uvicorn.run(
        "pc.api_server:app",
        host=host,
        port=port,
        log_level="info",
    )
    return 0


def _run_ui_mode(host: str, port: int) -> int:
    from streamlit.web import cli as stcli

    script_path = _resource_root() / "pc" / "app_streamlit.py"
    if not script_path.exists():
        raise FileNotFoundError(f"Missing Streamlit script: {script_path}")

    # Packaged apps must run with development mode off when forcing server.port.
    os.environ["STREAMLIT_GLOBAL_DEVELOPMENT_MODE"] = "false"

    sys.argv = [
        "streamlit",
        "run",
        str(script_path),
        "--global.developmentMode",
        "false",
        "--server.address",
        host,
        "--server.port",
        str(port),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]
    return int(stcli.main())


def _run_parent_mode(
    api_host: str,
    api_port: int,
    ui_host: str,
    ui_port: int,
    no_browser: bool,
    wait_timeout_s: float,
) -> int:
    cmd = _launcher_command()
    env = os.environ.copy()

    api_proc = subprocess.Popen(
        cmd + ["--mode", "api", "--api-host", api_host, "--api-port", str(api_port)],
        env=env,
    )
    ui_proc = subprocess.Popen(
        cmd + ["--mode", "ui", "--ui-host", ui_host, "--ui-port", str(ui_port)],
        env=env,
    )

    def _shutdown_children() -> None:
        _terminate_process(ui_proc)
        _terminate_process(api_proc)

    atexit.register(_shutdown_children)

    ui_url = f"http://{ui_host}:{ui_port}"
    if _wait_for_http(ui_url, wait_timeout_s) and not no_browser:
        webbrowser.open(ui_url)

    try:
        return int(ui_proc.wait())
    finally:
        _shutdown_children()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QTrim desktop launcher")
    parser.add_argument("--mode", choices=["parent", "api", "ui"], default="parent")
    parser.add_argument("--api-host", default="127.0.0.1")
    parser.add_argument("--api-port", type=int, default=8000)
    parser.add_argument("--ui-host", default="127.0.0.1")
    parser.add_argument("--ui-port", type=int, default=8501)
    parser.add_argument("--wait-timeout", type=float, default=30.0)
    parser.add_argument("--no-browser", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.mode == "api":
        return _run_api_mode(args.api_host, args.api_port)
    if args.mode == "ui":
        return _run_ui_mode(args.ui_host, args.ui_port)
    return _run_parent_mode(
        api_host=args.api_host,
        api_port=args.api_port,
        ui_host=args.ui_host,
        ui_port=args.ui_port,
        no_browser=args.no_browser,
        wait_timeout_s=args.wait_timeout,
    )


if __name__ == "__main__":
    raise SystemExit(main())
