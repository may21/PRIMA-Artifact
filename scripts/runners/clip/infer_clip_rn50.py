import os
import time
import clip
import torch
from torchvision.datasets import CIFAR100

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

if device == "cuda":
    print("cuda device:", torch.cuda.get_device_name(0))

# RN50 model
model, preprocess = clip.load('RN50', device=device, download_root='.')
model.eval()

# Load the dataset from local directory
cifar100 = CIFAR100(root=".", download=False, train=False)

# Prepare text inputs: CIFAR-100 class prompts
text_inputs = torch.cat([
    clip.tokenize(f"a photo of a {c}") for c in cifar100.classes
]).to(device)

print("model: RN50")
print("num_text_prompts:", len(cifar100.classes))
print("text_inputs shape:", tuple(text_inputs.shape))

# Select all images from CIFAR-100 test set
image_indices = list(range(10000))

total_time = 0.0

with torch.no_grad():
    for idx in image_indices:
        image, class_id = cifar100[idx]
        image_input = preprocess(image).unsqueeze(0).to(device)

        if device == "cuda":
            torch.cuda.synchronize()

        start = time.time()

        # CLIP-Full inference path:
        # image encoder + text encoder + similarity
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        values, top_indices = similarity[0].topk(5)

        if device == "cuda":
            torch.cuda.synchronize()

        end = time.time()
        latency_ms = (end - start) * 1000.0
        total_time += end - start

        print(f"\nImage index: {idx}")
        print(f"GT label: {cifar100.classes[class_id]}")
        print(f"Latency: {latency_ms:.3f} ms")
        print("Top predictions:")

        for value, pred_idx in zip(values, top_indices):
            print(f"{cifar100.classes[pred_idx]:>16s}: {100 * value.item():.2f}%")

print("\n===== Summary =====")
print(f"model: RN50")
print(f"num_images: {len(image_indices)}")
print(f"total_inference_time_sec: {total_time:.3f}")
print(f"avg_latency_ms: {(total_time / len(image_indices)) * 1000.0:.3f}")
