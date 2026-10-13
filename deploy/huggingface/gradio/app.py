"""Entry point for a Hugging Face *Gradio* Space (free on ZeroGPU hardware).

OreSight is a FastAPI app, not a Gradio app. A Gradio Space simply runs this file and expects a web
server on port 7860, so this file restores the saved workspace, launches a small Gradio page on a
side port (ZeroGPU needs a Gradio launch) and serves the OreSight app on the public port.
"""

import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
# Gradio's server-side rendering starts a Node helper that takes port 7860 before our server can
os.environ["GRADIO_SSR_MODE"] = "False"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "geoscan"))

import gradio as gr  # noqa: E402
import spaces  # noqa: E402
import uvicorn  # noqa: E402


@spaces.GPU
def gpu_check():
    """ZeroGPU Spaces need at least one GPU-decorated function. OreSight itself runs on the CPU."""
    return "ok"


from geoscan.webapp import persist  # noqa: E402

if persist.enabled():
    print(f"Restoring saved data from {persist.repo()} ...", flush=True)
    try:
        print(f"{persist.restore()} files in the workspace.", flush=True)
    except Exception as e:  # start anyway; the app then shows no areas
        print(f"WARNING: could not restore saved data: {e}", flush=True)

from geoscan.webapp.server import app as geoscan_app  # noqa: E402

with gr.Blocks(title="OreSight") as demo:
    gr.Markdown("**OreSight.** This Space serves the OreSight web app at its main address.")
    gr.Button("ZeroGPU check").click(gpu_check, outputs=gr.Textbox(label="status"))

if __name__ == "__main__":
    # ZeroGPU sends its start-up report from Gradio's launch(), so launch the small Gradio page on a side
    # port (not reachable from outside), then serve OreSight itself on the Space's public port.
    demo.launch(server_name="127.0.0.1", server_port=7861, prevent_thread_lock=True, ssr_mode=False,
                show_error=True)
    uvicorn.run(geoscan_app, host="0.0.0.0", port=7860, log_level="warning")
