import requests
import torch
from PIL import Image
from transformers import pipeline

# image_url = "https://llava-vl.github.io/static/images/view.jpg"
# image = Image.open(requests.get(image_url, stream=True).raw)
image = Image.open("universe.png")

model_id = "llava-hf/llava-1.5-7b-hf"
# model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
device = 0 if torch.cuda.is_available() else -1
pipe = pipeline("image-to-text", model=model_id, device = device)

max_new_tokens = 512
prompt = "USER: <image>\nwhat is this?\nAssistant:"
# prompt = "User: <image>\nFor the following questions, generate 5 CREATIVE and UNIQUE ideas with detailed explanations.\nQuestion : Just suppose you could become any animal. Which would you choose and why?\nAssistant: 1."

outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": max_new_tokens})

print(outputs[0]["generated_text"])