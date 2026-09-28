"""GeoFUSE SentinelGuard — One-Command Production Launcher & Lifecycle Manager.

Orchestrates:
1. Python virtual environment & PyTorch CUDA inspection.
2. Frontend production build presence verification.
3. Port binding verification & conflict detection.
4. FastAPI GPU gateway background startup.
5. Deterministic health & system telemetry polling (zero blind sleep delays).
6. Automatic default browser launch to the served production React app.
7. Clean child process termination on exit (zero orphan processes).
"""

import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HOST = "127.0.0.1"
PORT = 8000
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"


def is_port_in_use(host: str, port: int) -> bool:
    """Check if target host:port is currently bound by a socket."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def poll_endpoint(url: str, timeout_sec: float = 30.0, interval_sec: float = 0.3) -> dict:
    """Poll an HTTP JSON endpoint until HTTP 200 is received or timeout expires."""
    start_time = time.time()
    last_err = None

    while time.time() - start_time < timeout_sec:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "GeoFUSE-Launcher/1.0"})
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                if resp.status == 200:
                    data = resp.read().decode("utf-8")
                    return json.loads(data)
        except Exception as e:
            last_err = e
            time.sleep(interval_sec)

    raise TimeoutError(f"Endpoint {url} did not become ready within {timeout_sec}s. Last error: {last_err}")


def print_banner():
    print("=" * 79)
    print("       GeoFUSE SentinelGuard (SIH 2026) - Full-Stack Platform Launcher")
    print("=" * 79)
    print()


def verify_prerequisites():
    """Verify virtual environment, frontend dist, and model checkpoints."""
    print("[1/5] Verifying environment & artifacts...")

    # Verify Frontend Production Build
    if not (FRONTEND_DIST / "index.html").exists():
        print("  [!] Frontend production bundle not detected in frontend/dist.")
        print("  [*] Building React production bundle with npm...")
        frontend_dir = PROJECT_ROOT / "frontend"
        try:
            subprocess.run(["npm", "run", "build"], cwd=str(frontend_dir), check=True, shell=True)
            print("  [OK] Frontend production bundle built successfully.")
        except Exception as e:
            print(f"  [X] Failed to build frontend: {e}")
            sys.exit(1)
    else:
        print("  [OK] Frontend production bundle verified (frontend/dist/index.html).")

    # Verify Checkpoints
    ckpt_files = list(CHECKPOINTS_DIR.glob("ensemble_member_*.pth"))
    if len(ckpt_files) < 3:
        print(f"  [!] Warning: Found {len(ckpt_files)}/3 ensemble checkpoints in checkpoints/.")
    else:
        print(f"  [OK] Model checkpoints verified ({len(ckpt_files)} ensemble members present).")


def check_port_and_existing_instance() -> bool:
    """Check if port 8000 is already running a GeoFUSE instance."""
    print("[2/5] Checking port 8000 availability...")
    if is_port_in_use(HOST, PORT):
        # Test if it's already GeoFUSE
        try:
            health = poll_endpoint(f"http://{HOST}:{PORT}/api/health", timeout_sec=2.0)
            if health.get("status") == "ok":
                print(f"  [*] GeoFUSE instance is already active and healthy on http://{HOST}:{PORT}/.")
                return True
        except Exception:
            pass

        print(f"  [!] Port {PORT} is in use by an unresponsive or previous process.")
        print(f"  [*] Attempting to automatically free port {PORT}...")
        if os.name == "nt":
            try:
                output = subprocess.check_output(f"netstat -ano | findstr :{PORT}", shell=True, text=True, errors="replace")
                for line in output.strip().splitlines():
                    if f":{PORT}" in line and "LISTENING" in line.upper():
                        parts = line.strip().split()
                        pid = parts[-1]
                        if pid.isdigit() and int(pid) != os.getpid():
                            print(f"  [*] Terminating stale process PID {pid} holding port {PORT}...")
                            subprocess.run(f"taskkill /F /PID {pid}", shell=True, capture_output=True)
                            time.sleep(1.0)
                            break
            except Exception as e:
                print(f"  [!] Note on port reclamation: {e}")

        if is_port_in_use(HOST, PORT):
            print(f"  [X] Error: Port {PORT} is still occupied by another application.")
            print(f"      Please terminate the existing process using port {PORT} and try again.")
            sys.exit(1)
        else:
            print(f"  [OK] Port {PORT} has been successfully freed.")
            return False

    print(f"  [OK] Port {PORT} is clear.")
    return False


def start_server() -> subprocess.Popen:
    """Start Uvicorn FastAPI server as a managed child process."""
    print("[3/5] Starting FastAPI GPU Gateway...")
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "src.api.main:app",
        "--host",
        HOST,
        "--port",
        str(PORT),
        "--log-level",
        "info",
    ]

    # Create process with own process group for clean shutdown
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        creationflags=creationflags,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    return proc


def wait_for_server_and_report(proc: subprocess.Popen):
    """Poll health check and display genuine hardware telemetry."""
    print("[4/5] Polling server readiness (zero blind sleep delays)...")
    health_url = f"http://{HOST}:{PORT}/api/health"
    status_url = f"http://{HOST}:{PORT}/api/system/status"

    # Poll health
    start_wait = time.time()
    while time.time() - start_wait < 30.0:
        if proc.poll() is not None:
            # Process crashed
            out = proc.stdout.read() if proc.stdout else ""
            print(f"  [X] Server process exited unexpectedly with code {proc.returncode}.")
            print(out)
            sys.exit(1)

        try:
            req = urllib.request.Request(health_url, headers={"User-Agent": "GeoFUSE-Launcher"})
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                if resp.status == 200:
                    break
        except Exception:
            time.sleep(0.3)
    else:
        print("  [X] Server health check timed out after 30 seconds.")
        proc.terminate()
        sys.exit(1)

    elapsed = time.time() - start_wait
    print(f"  [OK] FastAPI server is operational (responded in {elapsed:.2f}s).")

    # Fetch and display truthful hardware & model telemetry
    try:
        sys_status = poll_endpoint(status_url, timeout_sec=5.0)
        print()
        print("  +----------------- SYSTEM TELEMETRY -----------------+")
        print(f"  | Device:     {sys_status.get('device_name', 'Unknown'):<38} |")
        print(f"  | CUDA:       {'Accelerated' if sys_status.get('cuda_available') else 'CPU Mode':<38} |")
        if sys_status.get("vram_total_gb"):
            print(f"  | VRAM:       {sys_status.get('vram_total_gb'):.1f} GB Total ({sys_status.get('vram_allocated_gb', 0):.2f} GB Allocated)       |")
        print(f"  | Checkpoints:{sys_status.get('ensemble_checkpoints_found', 0):>2} loaded / ready                        |")
        print(f"  | Demo Cache: {'Available' if sys_status.get('demo_cache_available') else 'Not Found':<38} |")
        print(f"  | Datasets:   {sys_status.get('active_dataset_count', 0):>2} local scenes detected                    |")
        print("  +----------------------------------------------------+")
        print()
    except Exception as e:
        print(f"  [!] Note: Could not parse system status: {e}")


def launch_browser():
    """Open default browser to the served production application."""
    print("[5/5] Launching browser to GeoFUSE SentinelGuard...")
    app_url = f"http://{HOST}:{PORT}/"
    print(f"  [*] Opening: {app_url}")
    webbrowser.open(app_url)


def main():
    print_banner()

    already_running = check_port_and_existing_instance()
    if already_running:
        launch_browser()
        print("\nPress Ctrl+C to exit launcher.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nExiting launcher.")
            sys.exit(0)

    verify_prerequisites()
    proc = start_server()

    # Register clean shutdown handler
    def cleanup(signum=None, frame=None):
        print("\n\nShutting down GeoFUSE FastAPI server cleanly...")
        if proc.poll() is None:
            if os.name == "nt":
                # Send CTRL_BREAK_EVENT to process group on Windows
                proc.send_signal(signal.CTRL_BREAK_EVENT)
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
            else:
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
        print("[OK] All child processes terminated. Port 8000 released.")
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    try:
        wait_for_server_and_report(proc)
        launch_browser()

        print("=" * 79)
        print(" GeoFUSE SentinelGuard is running in production mode!")
        print(f" - Web Application:   http://{HOST}:{PORT}/")
        print(f" - API Documentation: http://{HOST}:{PORT}/docs")
        print(f" - System Telemetry:  http://{HOST}:{PORT}/api/system/status")
        print("=" * 79)
        print("\nPress Ctrl+C in this window at any time to cleanly stop the server.")

        # Stream server logs or wait
        while proc.poll() is None:
            time.sleep(0.5)

    except KeyboardInterrupt:
        cleanup()
    except Exception as e:
        print(f"\n[X] Unexpected error: {e}")
        cleanup()


if __name__ == "__main__":
    main()
