"""MD6329 LecTrace lecture: one authentic BloodMNIST CNN training step.

Run: lectrace run bloodmnist_lecture.py
View: lectrace serve bloodmnist_lecture.py --port 8766
Requires: pip install lectrace torch torchvision medmnist
"""
from lectrace import text, note
import random
import numpy as np
import torch
from torch import nn
from torchvision import transforms
from medmnist import BloodMNIST

CLASS_NAMES = [
    "Basophil", "Eosinophil", "Erythroblast", "Immature granulocyte",
    "Lymphocyte", "Monocyte", "Neutrophil", "Platelet"
]


class SmallCNN(nn.Module):
    """Exact architecture from the submitted blood_cell notebook."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, 8)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def load_example():
    """Prepare one real training example, original and augmented."""
    original = BloodMNIST(split="train", transform=transforms.ToTensor(), download=True)
    augmented = BloodMNIST(
        split="train",
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
        ]),
        download=True,
    )
    image, label = original[0]
    aug_image, aug_label = augmented[0]
    return image, aug_image, int(np.asarray(label).reshape(-1)[0]), int(np.asarray(aug_label).reshape(-1)[0])


def single_training_step(model, image, label):
    """Run one optimizer step, returning compact values suitable for tracing."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    model.train()
    inputs = image.unsqueeze(0)
    targets = torch.tensor([label], dtype=torch.long)
    before = model.classifier[-1].weight.detach().clone()
    optimizer.zero_grad()
    logits = model(inputs)
    loss = criterion(logits, targets)
    loss.backward()
    grad_norm = float(model.classifier[-1].weight.grad.norm().item())
    optimizer.step()
    after = model.classifier[-1].weight.detach()
    change_norm = float((after - before).norm().item())
    return logits.detach().flatten().tolist(), float(loss.item()), grad_norm, change_norm


# Construct the PyTorch module before LecTrace begins tracing main().
# This avoids tracing nn.Module initialization internals.
DEMO_MODEL = SmallCNN()

def main():
    text("# BloodMNIST: How Mild Augmentation Affects CNN Training")
    text("MD6329 | Executable walkthrough of a single real training example")
    note("Research question: Does mild data augmentation improve CNN performance on blood-cell image classification?")

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # TRACE 1 — Input and label
    image, augmented_image, label, augmented_label = load_example()
    image_shape = tuple(image.shape)  # @inspect image_shape
    original_range = (round(float(image.min()), 3), round(float(image.max()), 3))  # @inspect original_range
    class_name = CLASS_NAMES[label]  # @inspect class_name
    text("## 1. Data input")
    note("A BloodMNIST image has 3 RGB channels and 28×28 pixels. ToTensor converts pixels to [0, 1].")

    # TRACE 2 — Augmentation (training only)
    augmented_shape = tuple(augmented_image.shape)  # @inspect augmented_shape
    label_unchanged = label == augmented_label  # @inspect label_unchanged
    pixel_difference = round(float((image - augmented_image).abs().mean().item()), 5)  # @inspect pixel_difference
    text("## 2. Mild data augmentation")
    note("The training pipeline uses random horizontal flipping (p=0.5) and rotation within ±15°. The class label remains unchanged. Validation and test data are NOT augmented.")

    # TRACE 3 — CNN architecture and forward pass
    model = DEMO_MODEL
    model.eval()
    with torch.no_grad():
        feature_shape = tuple(model.features(image.unsqueeze(0)).shape)  # @inspect feature_shape
        initial_logits = model(image.unsqueeze(0)).flatten().tolist()
    output_classes = len(initial_logits)  # @inspect output_classes
    text("## 3. CNN forward pass")
    note("The three convolutional blocks map 3→32→64→128 channels. Adaptive pooling yields [1,128,1,1], and the classifier produces eight logits.")

    # TRACE 4 — One real training update (not full experiment training)
    logits, loss_value, gradient_norm, weight_change_norm = single_training_step(
        model, augmented_image, label
    )
    loss_value = round(loss_value, 5)  # @inspect loss_value
    gradient_norm = round(gradient_norm, 5)  # @inspect gradient_norm
    weight_change_norm = round(weight_change_norm, 7)  # @inspect weight_change_norm
    text("## 4. Cross-entropy and backpropagation")
    note("loss.backward() computes gradients. optimizer.step() changes the weights. This is ONE demonstration step, not the 15-epoch/three-seed experiment.")

    # TRACE 5 — Interpret logits, not a validated performance result
    probabilities = torch.softmax(torch.tensor(logits), dim=0)
    predicted_index = int(probabilities.argmax().item())  # @inspect predicted_index
    predicted_name = CLASS_NAMES[predicted_index]  # @inspect predicted_name
    predicted_probability = round(float(probabilities[predicted_index]), 4)  # @inspect predicted_probability
    text("## 5. Prediction and evaluation")
    note("A prediction after one update is only a mechanics demonstration; it is NOT the reported test accuracy or Macro-F1. The report compares separately trained CNNs across three seeds.")
    text("Full-experiment mean test Macro-F1: CNN 0.8577; CNN + augmentation 0.8876 (gain +0.0299). Results vary by seed and class.")


if __name__ == "__main__":
    main()
