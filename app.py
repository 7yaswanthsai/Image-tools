import os
import gradio as gr
import torch
import numpy as np
import requests
import cv2
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry
from diffusers import AutoPipelineForText2Image, StableDiffusionXLControlNetPipeline, ControlNetModel
from huggingface_hub import login


# Hugging Face Token from Environment Variable
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("Please set HF_TOKEN as an environment variable")

login(token=HF_TOKEN)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32


# ========== Background Removal (SAM) ==========

MODEL_PATH = "sam_vit_h_4b8939.pth"
MODEL_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

if not os.path.exists(MODEL_PATH):
    print("Downloading SAM Model...")
    response = requests.get(MODEL_URL, stream=True)
    with open(MODEL_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Download completed!")

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
    output = image_np.copy()
    output[~mask] = [0, 0, 0]
    return Image.fromarray(output)


# ========== Text-to-Image (SDXL Turbo) ==========

text_pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo",
    torch_dtype=dtype,
    use_safetensors=True,
    requires_safety_checker=False
).to(device)

def generate_image(prompt):
    image = text_pipe(prompt, num_inference_steps=1, guidance_scale=0.0).images[0]
    return image


# ========== Style Transfer (SDXL + ControlNet) ==========

controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0",
    torch_dtype=dtype
).to(device)

style_pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    torch_dtype=dtype
).to(device)

def apply_style(image: Image.Image, prompt: str) -> Image.Image:
    image = image.resize((512, 512))
    img_np = np.array(image)
    img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(img_gray, 100, 200)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    edge_image = Image.fromarray(edges)

    result = style_pipe(prompt=prompt, image=edge_image, num_inference_steps=5).images[0]
    return result


# ========== Gradio Interface ==========

with gr.Blocks(title="AI Image Toolkit") as demo:
    gr.Markdown("## 🎨 AI Image Toolkit — Background Removal | Text-to-Image | Style Transfer")

    with gr.Tab("Background Removal"):
        gr.Markdown("### Upload an image to remove background")
        img_input = gr.Image(type="pil")
        img_output = gr.Image(type="pil")
        btn = gr.Button("Remove Background")
        btn.click(remove_background, inputs=img_input, outputs=img_output)

    with gr.Tab("Text to Image"):
        gr.Markdown("### Enter a prompt to generate an image")
        prompt_input = gr.Textbox(label="Prompt")
        output_image = gr.Image(type="pil")
        btn2 = gr.Button("Generate Image")
        btn2.click(generate_image, inputs=prompt_input, outputs=output_image)

    with gr.Tab("Style Transfer (ControlNet)"):
        gr.Markdown("### Upload an image and enter style prompt")
        style_img_input = gr.Image(type="pil")
        style_prompt_input = gr.Textbox(label="Style Prompt")
        styled_output = gr.Image(type="pil")
        btn3 = gr.Button("Apply Style")
        btn3.click(apply_style, inputs=[style_img_input, style_prompt_input], outputs=styled_output)

demo.launch()
