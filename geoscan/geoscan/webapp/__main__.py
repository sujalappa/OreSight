"""python -m geoscan.webapp [--port 8765] [--host 127.0.0.1]

If the port is taken by another program, the next free port is used. The browser opens only
once the server answers.
"""

import argparse
import socket
import threading
import time
import urllib.request
import webbrowser

import uvicorn


def free_port(host: str, start: int, tries: int = 20) -> int:
    for port in range(start, start + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, port))
                return port
            except OSError:
                print(f"Port {port} is in use by another program, trying {port + 1} ...")
    raise SystemExit(f"No free port between {start} and {start + tries - 1}")


def open_when_ready(url: str, timeout: float = 120.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(url + "/api/config", timeout=2)
            webbrowser.open(url)
            return
        except Exception:
            time.sleep(0.5)
    print("The server did not start in time - see the messages above.")


def main():
    ap = argparse.ArgumentParser(prog="geoscan.webapp")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    from . import persist
    if persist.enabled():
        print(f"  Restoring saved data from {persist.repo()} ...", flush=True)
        try:
            print(f"  {persist.restore()} files in the workspace.", flush=True)
        except Exception as e:
            print(f"  WARNING: could not restore saved data: {e}", flush=True)
    port = free_port(a.host, a.port)
    url = f"http://{a.host}:{port}"
    print(f"\n  OreSight is starting at {url}\n  (keep this window open; close it to stop the app)\n", flush=True)
    if not a.no_browser:
        threading.Thread(target=open_when_ready, args=(url,), daemon=True).start()
    uvicorn.run("geoscan.webapp.server:app", host=a.host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
