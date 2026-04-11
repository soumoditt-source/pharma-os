from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from typing import Dict

import requests


ROOT = Path(__file__).resolve().parents[1]
PID_FILE = ROOT / ".server.pid"
STDOUT_LOG = ROOT / ".server.out.log"
STDERR_LOG = ROOT / ".server.err.log"
START_LINE_RE = re.compile(r"^\[START\] task=\S+ env=\S+ model=.+$")
STEP_LINE_RE = re.compile(
    r"^\[STEP\] step=\d+ action=.* reward=-?\d+\.\d{2} done=(true|false) error=.*$"
)
END_LINE_RE = re.compile(
    r"^\[END\] success=(true|false) steps=\d+ score=-?\d+\.\d{2} rewards=-?\d+\.\d{2}(,-?\d+\.\d{2})*$"
)


def run(command: list[str], env: Dict[str, str] | None = None) -> None:
    print(f"[preflight] {' '.join(command)}")
    subprocess.run(command, cwd=ROOT, check=True, env=env)


def run_capture(command: list[str], env: Dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    print(f"[preflight] {' '.join(command)}")
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        env=env,
        capture_output=True,
        text=True,
    )
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    return completed


def validate_inference_stdout(stdout: str) -> None:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("inference.py emitted no stdout lines.")

    task_block_state = "expect_start"
    saw_end = False

    for line in lines:
        if START_LINE_RE.fullmatch(line):
            if task_block_state != "expect_start":
                raise RuntimeError(f"Unexpected [START] line order in inference output: {line}")
            task_block_state = "expect_step_or_end"
            saw_end = False
            continue

        if STEP_LINE_RE.fullmatch(line):
            if task_block_state != "expect_step_or_end":
                raise RuntimeError(f"Unexpected [STEP] line order in inference output: {line}")
            continue

        if END_LINE_RE.fullmatch(line):
            if task_block_state != "expect_step_or_end":
                raise RuntimeError(f"Unexpected [END] line order in inference output: {line}")
            task_block_state = "expect_start"
            saw_end = True
            continue

        raise RuntimeError(
            "Inference output contained a non-compliant stdout line. "
            f"Only [START], [STEP], and [END] lines are allowed: {line}"
        )

    if task_block_state != "expect_start" or not saw_end:
        raise RuntimeError("Inference output ended before emitting a final [END] line.")


def server_healthy(base_url: str) -> bool:
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False


def safe_unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except PermissionError:
        # On Windows, a recently-terminated process can hold the PID file open
        # briefly. The next write will still replace the stale contents.
        pass


def stop_existing_server() -> None:
    if not PID_FILE.exists():
        return
    try:
        server_pid = int(PID_FILE.read_text(encoding="utf-8").strip())
    except ValueError:
        safe_unlink(PID_FILE)
        return

    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(server_pid), "/F"], check=False, capture_output=True)
        else:
            os.kill(server_pid, 15)
    except OSError:
        pass
    safe_unlink(PID_FILE)


def start_server(host: str, port: int) -> subprocess.Popen:
    stop_existing_server()
    stdout_handle = STDOUT_LOG.open("w", encoding="utf-8")
    stderr_handle = STDERR_LOG.open("w", encoding="utf-8")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "server.app:app",
            "--host",
            host,
            "--port",
            str(port),
            "--workers",
            "1",
        ],
        cwd=ROOT,
        stdout=stdout_handle,
        stderr=stderr_handle,
    )
    PID_FILE.write_text(str(process.pid), encoding="utf-8")
    return process


def wait_for_server(base_url: str, timeout_s: int = 60) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if server_healthy(base_url):
            return
        time.sleep(1)
    raise RuntimeError(f"Server did not become healthy at {base_url} within {timeout_s} seconds.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PharmaOS local preflight checks.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--install", action="store_true", help="Install dependencies before validation.")
    parser.add_argument("--keep-server", action="store_true", help="Keep the local server running after checks.")
    parser.add_argument("--open-browser", action="store_true", help="Open the dashboard after checks.")
    parser.add_argument("--skip-inference", action="store_true", help="Skip the inference smoke test.")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    started_process: subprocess.Popen | None = None

    try:
        if args.install:
            run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
            run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

        run([sys.executable, "-m", "pytest", "-q"])
        run([sys.executable, "-m", "openenv.cli", "validate", "."])

        if not server_healthy(base_url):
            started_process = start_server(args.host, args.port)
            wait_for_server(base_url)

        run([sys.executable, "-m", "openenv.cli", "validate", "--url", base_url])

        if not args.skip_inference:
            inference_env = os.environ.copy()
            inference_env.setdefault("API_KEY", "dummy")
            inference_env.setdefault("HF_TOKEN", "dummy")
            inference_env.setdefault("API_BASE_URL", "http://127.0.0.1:9/v1")
            inference_env.setdefault("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
            inference_env["PHARMAO_URL"] = base_url
            inference_run = run_capture([sys.executable, "inference.py"], env=inference_env)
            validate_inference_stdout(inference_run.stdout)

        if args.open_browser:
            webbrowser.open(f"{base_url}/web")

        print(f"[preflight] Dashboard ready at {base_url}/web")
    finally:
        if started_process is not None and not args.keep_server:
            started_process.terminate()
            try:
                started_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                started_process.kill()
            safe_unlink(PID_FILE)


if __name__ == "__main__":
    main()
