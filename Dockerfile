# Worker RunPod Serverless: UltraTex (texturizado de mallas) sobre ComfyUI headless.
# Imagen completa: dependencias + ComfyUI + nodos UltraTex + pesos (~20 GB de pesos).
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    NVIDIA_DRIVER_CAPABILITIES=all \
    COMFY_DIR=/comfyui

# build-essential: Triton compila el kernel de atención dispersa de UltraTex al primer uso.
# libegl*: moderngl renderiza las vistas de normales sin pantalla (contexto EGL).
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl aria2 build-essential libgl1 libegl1 libegl-mesa0 libgles2 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ARG COMFY_COMMIT=52f98af2e2e42c421070a3e147c161c47cdeaf22
ARG ULTRATEX_COMMIT=d57e561d2911583d655b1b4091751a04647f2536

RUN git clone https://github.com/comfyanonymous/ComfyUI ${COMFY_DIR} \
    && git -C ${COMFY_DIR} checkout ${COMFY_COMMIT} \
    && pip install --no-cache-dir -r ${COMFY_DIR}/requirements.txt \
    && git clone https://github.com/visualbruno/ComfyUI-UltraTex ${COMFY_DIR}/custom_nodes/ComfyUI-UltraTex \
    && git -C ${COMFY_DIR}/custom_nodes/ComfyUI-UltraTex checkout ${ULTRATEX_COMMIT} \
    && pip install --no-cache-dir -r ${COMFY_DIR}/custom_nodes/ComfyUI-UltraTex/requirements.txt \
    && pip install --no-cache-dir runpod==1.7.* requests

# Sin pantalla: moderngl necesita un contexto EGL (por defecto busca X11 y falla).
RUN sed -i "s/moderngl.create_standalone_context(require=330)/moderngl.create_standalone_context(require=330, backend='egl')/" \
        ${COMFY_DIR}/custom_nodes/ComfyUI-UltraTex/ultratex_core/render.py \
    && grep -q "backend='egl'" ${COMFY_DIR}/custom_nodes/ComfyUI-UltraTex/ultratex_core/render.py

# Pesos. FLUX.2 Klein base 9B (fp8) desde el espejo del autor de los nodos: el repo
# oficial de Black Forest Labs pide aceptar la licencia (no comercial).
ARG HF=https://huggingface.co
ARG MS="https://www.modelscope.ai/api/v1/models/Yibo-Zhang/UltraTex/repo?Revision=master&FilePath="
RUN set -e; M=${COMFY_DIR}/models; \
    get() { mkdir -p "$(dirname "$2")"; aria2c -q -x 16 -s 16 -d "$(dirname "$2")" -o "$(basename "$2")" "$1"; test -s "$2"; }; \
    get "${HF}/visualbruno/FLUX.2-klein-base-9b-fp8/resolve/main/flux-2-klein-base-9b-fp8.safetensors" $M/diffusion_models/flux-2-klein-base-9b-fp8.safetensors; \
    get "${HF}/Comfy-Org/vae-text-encorder-for-flux-klein-9b/resolve/main/split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors" $M/text_encoders/qwen_3_8b_fp8mixed.safetensors; \
    get "${HF}/Comfy-Org/vae-text-encorder-for-flux-klein-9b/resolve/main/split_files/vae/flux2-vae.safetensors" $M/vae/flux2-vae.safetensors; \
    get "${MS}flux2/lora/dit_lora.safetensors" $M/loras/ultratex_flux2.safetensors; \
    get "${MS}flux2_mr/lora/dit_lora.safetensors" $M/loras/ultratex_flux2_mr.safetensors; \
    get "${MS}flux2/decoder.pt" $M/ultratex/ultratex_flux2_decoder.pt

WORKDIR /app
COPY handler.py result_transport.py workflow.json ./
CMD ["python", "-u", "handler.py"]
