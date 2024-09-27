import os
import subprocess
from subprocess import getoutput

#ComfyUI_Repo
repo_url = "https://github.com/comfyanonymous/ComfyUI"
branch   = "master"

#Nodes
nodes_list = [
    "https://github.com/ltdrdata/ComfyUI-Manager",
    "https://github.com/Fannovel16/comfyui_controlnet_aux",
    "https://github.com/cubiq/ComfyUI_IPAdapter_plus",
    "https://github.com/ssitu/ComfyUI_UltimateSDUpscale",
    "https://github.com/Acly/comfyui-inpaint-nodes",
    "https://github.com/Acly/comfyui-tooling-nodes"
]

#Models
model_list = [
    "https://huggingface.co/cagliostrolab/animagine-xl-3.1/resolve/main/animagine-xl-3.1.safetensors"
]

#Vae
vae_list = [
    "https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/sdxl_vae.safetensors"
]

#Lora
lora_list = [
    "https://huggingface.co/Linaqruf/anime-detailer-xl-lora/resolve/main/anime-detailer-xl.safetensors",
    "https://huggingface.co/latent-consistency/lcm-lora-sdv1-5/resolve/main/pytorch_lora_weights.safetensors",
    "https://huggingface.co/latent-consistency/lcm-lora-sdxl/resolve/main/pytorch_lora_weights.safetensors",
    "https://huggingface.co/ByteDance/Hyper-SD/resolve/main/Hyper-SDXL-8steps-CFG-lora.safetensors"
]

#ControlNet
controlnet_list = [
    "https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_inpaint_fp16.safetensors",
    "https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_lora_rank128_v11f1e_sd15_tile_fp16.safetensors",
    "https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/thibaud_xl_openpose_256lora.safetensors"
]

#Upscaler
upscaler_list = [
    "https://huggingface.co/Kim2091/AnimeSharp/resolve/main/4x-AnimeSharp.pth",
    "https://huggingface.co/Kim2091/UltraSharp/resolve/main/4x-UltraSharp.pth",
    "https://huggingface.co/gemasai/4x_NMKD-Superscale-SP_178000_G/resolve/main/4x_NMKD-Superscale-SP_178000_G.pth",
    "https://huggingface.co/Acly/Omni-SR/resolve/main/OmniSR_X2_DIV2K.safetensors",
    "https://huggingface.co/Acly/Omni-SR/resolve/main/OmniSR_X3_DIV2K.safetensors",
    "https://huggingface.co/Acly/Omni-SR/resolve/main/OmniSR_X4_DIV2K.safetensors",
    "https://huggingface.co/KoRo8888/yandereneo/resolve/main/4x_NMKD-YandereNeoXL_200k.pth"
]

inpaint_list = [
    "https://huggingface.co/Acly/MAT/resolve/main/MAT_Places512_G_fp16.safetensors",
    "https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/fooocus_inpaint_head.pth",
    "https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint_v26.fooocus.patch"
]

ipadapter_list = [
    "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15.safetensors",
    "https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter_sdxl_vit-h.safetensors"
]

root_dir               = "/tmp"
comfyui_dir            = os.path.join(root_dir,      "ComfyUI")
nodes_dir              = os.path.join(comfyui_dir,   "custom_nodes")
models_dir             = os.path.join(comfyui_dir,   "models", "checkpoints")
vae_dir                = os.path.join(comfyui_dir,   "models", "vae")
lora_dir               = os.path.join(comfyui_dir,   "models", "loras")
controlnet_dir         = os.path.join(comfyui_dir,   "models", "controlnet")
upscaler_dir           = os.path.join(comfyui_dir,   "models", "upscale_models")
inpaint_dir            = os.path.join(comfyui_dir,   "models", "inpaint")
ipadapter_dir          = os.path.join(comfyui_dir,   "models", "ipadapter")

def clone_repo(branch, repo_url, comfyui_dir):
    subprocess.run(['git', 'clone', '-b', branch, repo_url, comfyui_dir])

def install_dependencies():
    subprocess.run(['apt-get', 'install', '-qq', 'lz4', '-y'])
    subprocess.run(['npm', 'install', '-g', 'localtunnel'])
    subprocess.run(['pip', 'install', 'aria2', 'gdown', 'torchsde', 'einops'])
    subprocess.run(['pip', 'install', 'onnxruntime-gpu', '--extra-index-url', 'https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/'])
    subprocess.run(['gdown', 'https://drive.google.com/uc?id=1G2Es3h34jrhoC1IEc3pVHEk53XMQjm-m', '-O', '/tmp/ComfyUI/tunnel'])
    subprocess.run(['curl', '-s', '-Lo', '/usr/bin/cl', 'https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64'])
    subprocess.run(['chmod', '+x', '/usr/bin/cl'])

def prepare_environment():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["SAFETENSORS_FAST_GPU"] = "1"
    os.environ["PYTHONWARNINGS"] = "ignore"
    os.environ["USE_TORCH_COMPILE"] = "1"

def download_file(url, output_dir):
    filename = url.split("/")[-1]
    if "lcm-lora-sdv1-5" in url:
        filename = "lcm-lora-sdv1-5.safetensors"
    elif "lcm-lora-sdxl" in url:
        filename = "lcm-lora-sdxl.safetensors"
    output_path = os.path.join(output_dir, filename)
    subprocess.run(['aria2c', '--console-log-level=error', '-c', '-x', '16', '-s', '16', '-k', '1M', url, '-d', output_dir, '-o', filename])

def clone_nodes(nodes_list, nodes_dir):
    for nodes_url in nodes_list:
        repo_name = nodes_url.split("/")[-1]
        repo_dir = os.path.join(nodes_dir, repo_name)
        clone_command = ['git', 'clone']
        if nodes_url == "https://github.com/ssitu/ComfyUI_UltimateSDUpscale":
            clone_command.append('--recursive')
        clone_command.extend([nodes_url, repo_dir])
        subprocess.run(clone_command)

def download_model(model_list, models_dir):
    for model_url in model_list:
        download_file(model_url, models_dir)

def download_vae(vae_list, vae_dir):
    for vae_url in vae_list:
        download_file(vae_url, vae_dir)

def download_lora(lora_list, lora_dir):
    for lora_url in lora_list:
        download_file(lora_url, lora_dir)

def download_controlnet(controlnet_list, controlnet_dir):
    for controlnet_url in controlnet_list:
        download_file(controlnet_url, controlnet_dir)

def download_upscaler(upscaler_list, upscaler_dir):
    for upscaler_url in upscaler_list:
        download_file(upscaler_url, upscaler_dir)

def download_inpaint(inpaint_list, inpaint_dir):
    for inpaint_url in inpaint_list:
        download_file(inpaint_url, inpaint_dir)

def download_ipadapter(ipadapter_list, ipadapter_dir):
    for ipadapter_url in ipadapter_list:
        download_file(ipadapter_url, ipadapter_dir)

def download_shared_models():
    subprocess.run(['wget', '-q', 'https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors', '-P', f'{comfyui_dir}/models/clip_vision/SD1.5'])
    
def launch():
    import cloudpickle as pickle
    import re
    subprocess.run(f'echo "\033[1;36mYOUR WEBUI IP ADDRESS: $(curl -s ipv4.icanhazip.com)\033[0m\n"', shell=True)
    tunnel = pickle.load(open("tunnel", "rb"), encoding="utf-8")
    tunnel_port = 7860
    tunnel_list = [
        {
            "command": "cl tunnel --url localhost:{port}",
            "pattern": re.compile(r"[\w-]+\.trycloudflare\.com"),
        },
        {
            "command": "lt --port {port}",
            "pattern": re.compile(r"[\w-]+\.loca\.lt"),
        },
    ]
    with tunnel(tunnel_port, tunnel_list, debug=False):
        launch_command = [
            'python',
            'main.py',
            '--use-pytorch-cross-attention',
            '--listen',
            '--port', str(tunnel_port),
        ]
        subprocess.run(launch_command)

def main():
    prepare_environment()
    clone_repo(branch, repo_url, comfyui_dir)
    clone_nodes(nodes_list, nodes_dir)
    install_dependencies()
    download_model(model_list, models_dir)
    download_vae(vae_list, vae_dir)
    download_lora(lora_list, lora_dir)
    download_controlnet(controlnet_list, controlnet_dir)
    download_upscaler(upscaler_list, upscaler_dir)
    download_inpaint(inpaint_list, inpaint_dir)
    download_ipadapter(ipadapter_list, ipadapter_dir)
    download_shared_models()
    os.chdir(comfyui_dir)

main()
