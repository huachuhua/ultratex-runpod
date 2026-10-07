"""
handler.py — Worker Serverless de RunPod para UltraTex (SB-176)

Texturiza una malla sin textura a partir de una imagen de referencia: genera 6
vistas con FLUX.2 Klein base 9B + la LoRA de UltraTex y las hornea en una textura
UV (color y, opcionalmente, metálico/rugosidad). Devuelve un .GLB.

Corre ComfyUI sin interfaz dentro del contenedor (127.0.0.1:8188) y le manda el
workflow de `workflow.json` (formato API), con los nodos de
github.com/visualbruno/ComfyUI-UltraTex.
"""

import base64
import copy
import gzip
import io
import json
import os
import subprocess
import time
import traceback
import urllib.request
import uuid

import numpy as np
import runpod
from PIL import Image
from scipy import ndimage

from result_transport import stream_files

COMFY_DIR = os.environ.get("COMFY_DIR", "/comfyui")
COMFY_URL = "http://127.0.0.1:8188"
WORKFLOW = json.load(open(os.path.join(os.path.dirname(__file__), "workflow.json")))
INPUT_DIR = os.path.join(COMFY_DIR, "input")
OUTPUT_DIR = os.path.join(COMFY_DIR, "output")

RESOLUTIONS = (1024, 2048)
TEXTURE_SIZES = (1024, 2048, 4096)
UV_MODES = ("comfy_gpu", "xatlas", "keep_existing")
MAX_JOB_SECONDS = 1800

_comfy = None


def start_comfy():
    """Arranca ComfyUI una vez por worker y espera a que responda."""
    global _comfy
    if _comfy is not None and _comfy.poll() is None:
        return
    _comfy = subprocess.Popen(
        ["python", "main.py", "--listen", "127.0.0.1", "--port", "8188", "--disable-auto-launch"],
        cwd=COMFY_DIR,
    )
    deadline = time.time() + 300
    while time.time() < deadline:
        if _comfy.poll() is not None:
            raise RuntimeError(f"ComfyUI terminó al arrancar (código {_comfy.returncode})")
        try:
            urllib.request.urlopen(COMFY_URL + "/system_stats", timeout=5)
            return
        except Exception:
            time.sleep(2)
    raise RuntimeError("ComfyUI no respondió en 5 minutos")


def comfy(path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(COMFY_URL + path, data=data,
                                 headers={"Content-Type": "application/json"} if data else {})
    return json.loads(urllib.request.urlopen(req, timeout=60).read())


def decode_b64(raw):
    if "," in raw[:100] and raw.startswith("data:"):
        raw = raw.split(",", 1)[1]
    return base64.b64decode(raw)


def decode_mesh(raw, gzipped):
    data = decode_b64(raw)
    if gzipped:
        data = gzip.decompress(data)
    if data[:4] != b"glTF":
        raise ValueError("'mesh' tiene que ser un .glb (binario glTF)")
    return data


def decode_reference(raw):
    """Imagen con el objeto recortado. Sin transparencia, se quita el fondo liso que toca los bordes."""
    image = Image.open(io.BytesIO(decode_b64(raw)))
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        image = image.convert("RGBA")
        if np.asarray(image)[..., 3].min() < 250:
            return image
    rgb = np.asarray(image.convert("RGB")).astype(np.int16)
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    bg = np.median(border, axis=0)
    close = np.abs(rgb - bg).max(axis=2) <= 18
    labels, _ = ndimage.label(close)
    edge = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    background = np.isin(labels, edge[edge > 0])
    alpha = ndimage.binary_erosion(~background, iterations=1)
    if alpha.mean() < 0.01:
        raise ValueError("No se pudo separar el objeto del fondo: mandá la imagen con transparencia")
    rgba = np.dstack([rgb.astype(np.uint8), (alpha * 255).astype(np.uint8)])
    return Image.fromarray(rgba, "RGBA")


def parse_options(job_input):
    def pick(name, default, allowed):
        value = job_input.get(name, default)
        if value not in allowed:
            raise ValueError(f"'{name}' debe ser uno de {list(allowed)} (llegó {value!r})")
        return value

    opts = {
        "resolution": pick("resolution", 1024, RESOLUTIONS),
        "texture_size": pick("texture_size", 2048, TEXTURE_SIZES),
        "uv_mode": pick("uv_mode", "comfy_gpu", UV_MODES),
        "pbr": bool(job_input.get("pbr", True)),
        "seed": int(job_input.get("seed", 12345)),
        "steps": int(job_input.get("steps", 25)),
        "guidance": float(job_input.get("guidance", 4)),
    }
    if not 10 <= opts["steps"] <= 50:
        raise ValueError("'steps' debe estar entre 10 y 50")
    return opts


def build_prompt(mesh_path, ref_name, opts):
    p = copy.deepcopy(WORKFLOW)
    p["22"]["inputs"]["glb_path"] = mesh_path
    # keep_existing necesita las UV de la malla; los otros modos las rehacen.
    p["22"]["inputs"]["only_vertices_and_faces"] = opts["uv_mode"] != "keep_existing"
    p["19"]["inputs"]["image"] = ref_name
    p["2"]["inputs"]["uv_mode"] = opts["uv_mode"]
    p["12"]["inputs"]["texture_size"] = opts["texture_size"]
    p["12"]["inputs"]["filename_prefix"] = f"UltraTex/{uuid.uuid4().hex}"
    for sampler in ("4", "24"):
        p[sampler]["inputs"].update(resolution=opts["resolution"], seed=opts["seed"],
                                    steps=opts["steps"], guidance=opts["guidance"])
    # El mapa metálico/rugosidad es de baja frecuencia: 1024 alcanza (lo dice el README de UltraTex).
    p["24"]["inputs"]["resolution"] = 1024
    if not opts["pbr"]:
        for node in ("23", "24"):
            p.pop(node)
        p["12"]["inputs"].pop("metallic_roughness_views")
    return p


def run_prompt(prompt):
    pid = comfy("/prompt", {"prompt": prompt, "client_id": "runpod"})["prompt_id"]
    deadline = time.time() + MAX_JOB_SECONDS
    while time.time() < deadline:
        history = comfy(f"/history/{pid}")
        if pid in history:
            entry = history[pid]
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                for kind, msg in status.get("messages", []):
                    if kind == "execution_error":
                        tb = "".join(msg.get("traceback", [])[-3:])
                        raise RuntimeError(f"{msg.get('node_type')}: {msg.get('exception_message')} {tb}".strip())
                raise RuntimeError("ComfyUI falló sin mensaje")
            if status.get("completed"):
                return entry["outputs"]
        if _comfy.poll() is not None:
            raise RuntimeError(f"ComfyUI se cayó durante el trabajo (código {_comfy.returncode})")
        time.sleep(2)
    raise TimeoutError("UltraTex tardó más de 30 minutos")


def find_glb(outputs):
    for out in outputs.values():
        for item in out.get("3d", []):
            if item.get("filename", "").endswith(".glb"):
                return os.path.join(OUTPUT_DIR, item.get("subfolder", ""), item["filename"])
    raise RuntimeError("UltraTex terminó sin entregar el .glb")


def handler(job):
    """
    input:
      "mesh": base64 de un .glb (obligatorio)        "mesh_gzip": true si viene comprimido con gzip
      "image": base64/data URL de la referencia (obligatorio; ideal PNG con transparencia, de frente)
      "output_transport": "trellis-files-v1" (obligatorio)
      "resolution": 1024|2048   "texture_size": 1024|2048|4096   "uv_mode": comfy_gpu|xatlas|keep_existing
      "pbr": bool (metálico/rugosidad)   "seed", "steps", "guidance"
    """
    job_input = job.get("input") or {}
    if "transport_probe_bytes" in job_input:
        size = int(job_input["transport_probe_bytes"])
        if not 1 <= size <= 80 * 1024 * 1024:
            yield {"error": "transport_probe_bytes debe estar entre 1 y 80 MiB"}
            return
        yield from stream_files({"glb": os.urandom(size)})
        return
    if job_input.get("output_transport") != "trellis-files-v1":
        yield {"error": "Este worker solo entrega modelos por fragmentos (output_transport: trellis-files-v1)."}
        return
    if not job_input.get("mesh") or not job_input.get("image"):
        yield {"error": "Faltan 'mesh' y/o 'image' en el input."}
        return

    temp = []
    try:
        opts = parse_options(job_input)
        mesh = decode_mesh(job_input["mesh"], job_input.get("mesh_gzip", False))
        reference = decode_reference(job_input["image"])
    except Exception as e:
        yield {"error": str(e)}
        return

    try:
        start_comfy()
        name = uuid.uuid4().hex
        mesh_path = os.path.join(INPUT_DIR, f"{name}.glb")
        ref_name = f"{name}.png"
        os.makedirs(INPUT_DIR, exist_ok=True)
        with open(mesh_path, "wb") as f:
            f.write(mesh)
        reference.save(os.path.join(INPUT_DIR, ref_name))
        temp += [mesh_path, os.path.join(INPUT_DIR, ref_name)]

        t0 = time.time()
        outputs = run_prompt(build_prompt(mesh_path, ref_name, opts))
        glb_path = find_glb(outputs)
        temp.append(glb_path)
        with open(glb_path, "rb") as f:
            glb = f.read()
        print(f"UltraTex listo en {time.time() - t0:.0f}s, {len(glb) / 1e6:.1f} MB", flush=True)
        yield from stream_files({"glb": glb})
    except Exception as e:
        traceback.print_exc()
        yield {"error": f"{type(e).__name__}: {e}"[-3000:]}
    finally:
        for path in temp:
            try:
                os.remove(path)
            except OSError:
                pass


if __name__ == "__main__":
    start_comfy()
    runpod.serverless.start({"handler": handler, "return_aggregate_stream": False})
