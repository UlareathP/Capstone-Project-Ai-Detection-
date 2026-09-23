import os
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

DATA_DIR = "data/dataset/rvf10k"

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=transform)
valid_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "valid"), transform=transform)

print("Classes:", train_dataset.classes)
print("Train samples:", len(train_dataset))
print("Valid samples:", len(valid_dataset))

# Show a few sample images
fig, axes = plt.subplots(1, 4, figsize=(12, 3))
for i in range(4):
    img, label = train_dataset[i * 500]
    axes[i].imshow(img.permute(1, 2, 0))
    axes[i].set_title(train_dataset.classes[label])
    axes[i].axis("off")
plt.tight_layout()
plt.savefig("notebooks/sample_images.png")
print("Saved sample grid to notebooks/sample_images.png")