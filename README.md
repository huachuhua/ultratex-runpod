# Worker RunPod Serverless: UltraTex

Texturiza una **malla sin textura** a partir de una **imagen de referencia** con
[UltraTex](https://yiboz2001.github.io/UltraTex/) (SIGGRAPH Asia 2026): genera 6 vistas
(frente, izquierda, atrás, derecha, arriba, abajo) con FLUX.2 Klein base 9B + la LoRA de
UltraTex y las hornea en una textura UV. Opcionalmente predice metálico/rugosidad (ORM).

Corre ComfyUI sin interfaz dentro del contenedor con los nodos de
[visualbruno/ComfyUI-UltraTex](https://github.com/visualbruno/ComfyUI-UltraTex) y le manda
`workflow.json` (formato API). Es el cuarto motor de SB-176: no genera desde cero, repinta la
malla de un modelo existente (pensado para las de TRELLIS.2).

## Validación

Prueba de factibilidad del 2026-10-07 en un pod RTX 4090 con el mismo workflow: chibi de botas de
TRELLIS.2 (984 k caras), referencia de frente con transparencia, 1024 px → 283 s en total
(incluida la carga en frío), GLB con color y ORM de 2048². Metálico 0–0,33 en vez del 1,0 que
trae TRELLIS por omisión.

## Entrada (`input`)

| Campo | Default | Valores |
|---|---|---|
| `mesh` | — (obligatorio) | base64 de un `.glb` |
| `mesh_gzip` | `false` | `true` si `mesh` es gzip del `.glb` |
| `image` | — (obligatorio) | base64 o data URL. Mejor PNG con transparencia y de frente; sin transparencia se quita el fondo liso que toca los bordes |
| `output_transport` | — (obligatorio) | `trellis-files-v1` |
| `resolution` | `1024` | `1024` (~5 min) o `2048` (~4× más) |
| `texture_size` | `2048` | `1024`, `2048`, `4096` |
| `uv_mode` | `comfy_gpu` | `comfy_gpu`, `xatlas` (CPU, lento) o `keep_existing` (conserva las UV de la malla: sirve para reemplazar solo la imagen de un GLB rigueado) |
| `pbr` | `true` | `false` omite metálico/rugosidad |
| `seed`, `steps`, `guidance` | `12345`, `25`, `4` | |

RunPod acepta ~10 MB por trabajo: el backend de SB-176 manda la malla sin texturas y reducida
(`tools/preparar_malla_ultratex.py`, 200 k caras ≈ 4 MB en base64).

Salida por `/stream/{jobId}` con el protocolo `trellis-files-v1` (el mismo de los workers de
TRELLIS): un solo archivo, `glb`. Diagnóstico sin GPU: `input.transport_probe_bytes`.

## Modelos (horneados en la imagen, ~20 GB)

| Archivo | Origen | Licencia |
|---|---|---|
| `flux-2-klein-base-9b-fp8` | espejo `visualbruno/FLUX.2-klein-base-9b-fp8` (el oficial de BFL pide aceptar la licencia) | FLUX non-commercial |
| `qwen_3_8b_fp8mixed`, `flux2-vae` | `Comfy-Org/vae-text-encorder-for-flux-klein-9b` | |
| LoRAs `ultratex_flux2`, `ultratex_flux2_mr` y `ultratex_flux2_decoder.pt` | ModelScope `Yibo-Zhang/UltraTex` | |

⚠️ FLUX.2 Klein **9B** es de uso no comercial: sirve para uso personal; si algún día se vende
algo hecho con esto, revisar la licencia.

## Despliegue

1. Repo `huachuhua/ultratex-runpod` con esta carpeta. GitHub Actions publica
   `ghcr.io/huachuhua/ultratex-runpod`. **La primera vez** lanzar el workflow a mano con
   `full_build=1` (el `Dockerfile.handler` parte de esa imagen). Después, un push solo
   reemplaza handler y workflow (~5 min).
2. Endpoint Serverless: imagen fijada por `sha-<commit>` (no `latest`: los workers reutilizan
   la cacheada), **GPU RTX 4090 / 3090** (24 GB; Blackwell no), disco 50 GB, Workers Min 0,
   Idle Timeout 60 s, timeout de ejecución 30 min.
3. SB-176: pegar el ID del endpoint en ⚙️ Configurar → «RunPod UltraTex Endpoint ID».

## Trampas ya resueltas

- `moderngl.create_standalone_context()` busca X11: sin pantalla falla. El Dockerfile lo
  cambia a `backend='egl'` y verifica el reemplazo.
- La atención dispersa de UltraTex es un kernel de Triton que se compila al primer uso: la
  imagen trae `build-essential`.
- Un script de arranque con `exec > >(tee …)` seguido de `wait` se cuelga para siempre: `wait`
  también espera al `tee`. (Costó media hora en el pod de prueba.)
