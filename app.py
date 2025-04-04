import gradio as gr
import torch
import numpy as np
import requests
import os
from PIL import Image
import cv2

from segment_anything import SamPredictor, sam_model_registry
from diffusers import (
    AutoPipelineForText2Image,
    StableDiffusionXLControlNetPipeline,
    ControlNetModel
)
from huggingface_hub import login

# ------------------- SETUP -------------------

# 🔐 Hugging Face API Key
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("⚠️ 'HF_TOKEN' not set. Please set it as an environment variable.")
login(token=HF_TOKEN)

# ⚙️ Device & dtype
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# ------------------- BACKGROUND REMOVAL (SAM) -------------------

# 📥 Download SAM model if needed
MODEL_PATH = "sam_vit_h_4b8939.pth"
MODEL_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

if not os.path.exists(MODEL_PATH):
    print("⬇️ Downloading SAM model...")
    r = requests.get(MODEL_URL, stream=True)
    with open(MODEL_PATH, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

sam = sam_model_registry["vit_h"](checkpoint=MODEL_PATH).to(device)
sam_predictor = SamPredictor(sam)

def remove_background(image):
    image = image.convert("RGB")
    image_np = np.array(image)
    sam_predictor.set_image(image_np)

    h, w, _ = image_np.shape
    input_points = np.array([[w // 2, h // 2]])
    input_labels = np.array([1])
    masks, _, _ = sam_predictor.predict(input_points, input_labels, multimask_output=False)
    mask = masks[0]

    bg_removed = image_np.copy()
    bg_removed[~mask] = [0, 0, 0]
    return Image.fromarray(bg_removed)

# ------------------- TEXT TO IMAGE (SDXL TURBO) -------------------

text_pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo",
    torch_dtype=dtype,
    use_safetensors=True,
    requires_safety_checker=False
).to(device)

def generate_image(prompt):
    image = text_pipe(prompt, num_inference_steps=1, guidance_scale=0.0).images[0]
    return image

# ------------------- STYLE TRANSFER (SDXL + CONTROLNET) -------------------

controlnet = ControlNetModel.from_pretrained(
    "./models/controlnet-canny-sdxl-1.0",
    torch_dtype=torch.float32
)

style_pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "./models/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    torch_dtype=torch.float32
).to("cpu")  # style transfer runs on CPU

def apply_style(image: Image.Image, prompt: str) -> Image.Image:
    image = image.resize((512, 512))
    image_np = np.array(image)
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    canny = cv2.Canny(gray, 100, 200)
    canny_rgb = cv2.cvtColor(canny, cv2.COLOR_GRAY2RGB)
    edge_image = Image.fromarray(canny_rgb)
    result = style_pipe(prompt=prompt, image=edge_image, num_inference_steps=5).images[0]
    return result

# ------------------- GRADIO UI -------------------

with gr.Blocks(title="All-in-One AI Image Toolkit") as demo:
    gr.Markdown("## 🧠 All-in-One AI Image Toolkit")
    gr.Markdown("This app combines **Background Removal**, **Text-to-Image Generation**, and **Style Transfer** using advanced models like SAM, SDXL, and ControlNet.")

    with gr.Tabs():
        with gr.Tab("🖼️ Background Removal"):
            with gr.Row():
                input_bg = gr.Image(type="pil", label="Upload Image")
                output_bg = gr.Image(type="pil", label="Background Removed")
            btn_bg = gr.Button("Remove Background")
            btn_bg.click(fn=remove_background, inputs=input_bg, outputs=output_bg)

        with gr.Tab("🧾 Text-to-Image"):
            prompt_txt2img = gr.Textbox(label="Enter your prompt", placeholder="e.g. futuristic city at night")
            output_txt2img = gr.Image(type="pil", label="Generated Image")
            btn_txt2img = gr.Button("Generate Image")
            btn_txt2img.click(fn=generate_image, inputs=prompt_txt2img, outputs=output_txt2img)

        with gr.Tab("🎨 Style Transfer with SDXL + ControlNet"):
            with gr.Row():
                input_style_img = gr.Image(type="pil", label="Upload Image")
                prompt_style = gr.Textbox(label="Style Prompt", placeholder="e.g. anime style, Van Gogh")
            output_style = gr.Image(type="pil", label="Styled Image")
            btn_style = gr.Button("Apply Style")
            btn_style.click(fn=apply_style, inputs=[input_style_img, prompt_style], outputs=output_style)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, debug=True)
