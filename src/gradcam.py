import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms, models
import torch.nn.functional as F

DATA_DIR = "data/dataset/rvf10k/valid"
MODEL_PATH = "models/resnet18_deepfake_best.pt"
CLASS_NAMES = ["fake", "real"]  # alphabetical order, matches ImageFolder

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# --- Grad-CAM hooks on the last conv layer (layer4) ---
activations = {}
gradients = {}

def forward_hook(module, input, output):
    activations["value"] = output

def backward_hook(module, grad_input, grad_output):
    gradients["value"] = grad_output[0]

target_layer = model.layer4[-1]
target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)

def generate_gradcam(image_path):
    img = Image.open(image_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(device)

    output = model(input_tensor)
    pred_class = output.argmax(dim=1).item()

    model.zero_grad()
    output[0, pred_class].backward()

    grads = gradients["value"][0]       # [C, H, W]
    acts = activations["value"][0]      # [C, H, W]
    weights = grads.mean(dim=(1, 2))    # [C]

    cam = torch.zeros(acts.shape[1:], device=device)
    for i, w in enumerate(weights):
        cam += w * acts[i]
    cam = F.relu(cam)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    cam = cam.cpu().detach().numpy()

    cam_img = Image.fromarray(np.uint8(cam * 255)).resize((224, 224))
    cam_img = np.array(cam_img)

    img_resized = img.resize((224, 224))
    return img_resized, cam_img, CLASS_NAMES[pred_class]

# --- Pick 5 fake + 5 real random samples ---
N_PER_CLASS = 5
random.seed(42)  # same set every time, useful for comparing runs

sample_paths = []
sample_labels = []
for cls in ["fake", "real"]:
    folder = os.path.join(DATA_DIR, cls)
    files = os.listdir(folder)
    chosen = random.sample(files, N_PER_CLASS)
    sample_paths.extend([os.path.join(folder, f) for f in chosen])
    sample_labels.extend([cls] * N_PER_CLASS)

fig, axes = plt.subplots(4, N_PER_CLASS, figsize=(4 * N_PER_CLASS, 16))

for i in range(N_PER_CLASS):
    # Fake examples: row 0 = image, row 1 = Grad-CAM
    img, cam, pred_label = generate_gradcam(sample_paths[i])
    true_label = sample_labels[i]
    axes[0, i].imshow(img)
    axes[0, i].set_title(f"True: {true_label} | Pred: {pred_label}")
    axes[0, i].axis("off")
    axes[1, i].imshow(img)
    axes[1, i].imshow(cam, cmap="jet", alpha=0.5)
    axes[1, i].axis("off")

    # Real examples: row 2 = image, row 3 = Grad-CAM
    j = i + N_PER_CLASS
    img, cam, pred_label = generate_gradcam(sample_paths[j])
    true_label = sample_labels[j]
    axes[2, i].imshow(img)
    axes[2, i].set_title(f"True: {true_label} | Pred: {pred_label}")
    axes[2, i].axis("off")
    axes[3, i].imshow(img)
    axes[3, i].imshow(cam, cmap="jet", alpha=0.5)
    axes[3, i].axis("off")

plt.tight_layout()
os.makedirs("notebooks", exist_ok=True)
plt.savefig("notebooks/gradcam_examples.png", dpi=150)
print("Saved Grad-CAM visualization (10 examples) to notebooks/gradcam_examples.png")