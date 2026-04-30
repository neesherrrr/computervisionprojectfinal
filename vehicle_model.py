import copy
import json
import random
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATA_DIR = Path("vehicle-10")
OUTPUT_DIR = Path("outputs")
CLASS_NAMES = [
    "bicycle",
    "boat",
    "bus",
    "car",
    "helicopter",
    "minibus",
    "motorcycle",
    "taxi",
    "train",
    "truck",
]
IMAGE_SIZE = 128
BATCH_SIZE = 64
EPOCHS = 25
LEARNING_RATE = 0.001
TRAIN_SPLIT = 0.75
VAL_SPLIT = 0.15
TEST_SPLIT = 0.10
NUM_WORKERS = 0
SEED = 42
MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)
CAM_THRESHOLD = 0.6
LOCALIZATION_SAMPLES = 8


class Vehicle10Dataset(Dataset):
    def __init__(self, root, split, transform):
        self.root = Path(root)
        self.transform = transform

        meta_file = "train_meta.json" if split == "train" else "valid_meta.json"
        with open(self.root / meta_file, "r", encoding="utf-8") as file:
            meta = json.load(file)

        self.samples = list(zip(meta["path"], meta["label"]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        rel_path, label = self.samples[index]
        image = Image.open(self.root / rel_path).convert("RGB")
        image = self.transform(image)
        return image, label, rel_path


class SimpleVehicleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.35)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x, return_features=False):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        features = torch.relu(self.bn3(self.conv3(x)))
        x = self.gap(features)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        logits = self.fc(x)
        if return_features:
            return logits, features
        return logits


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            predictions = outputs.argmax(dim=1)

            total_loss += loss.item() * images.size(0)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)
            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

    return total_loss / total_samples, total_correct / total_samples, all_labels, all_predictions


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Classes: {CLASS_NAMES}")

    train_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )

    full_train_aug = Vehicle10Dataset(DATA_DIR, "train", train_transform)
    full_train_eval = Vehicle10Dataset(DATA_DIR, "train", eval_transform)

    train_counts = Counter(label for _, label in full_train_aug.samples)
    print("\nFull dataset summary:")
    for i, class_name in enumerate(CLASS_NAMES):
        print(f"  {class_name:12s}: {train_counts[i]}")

    indices = list(range(len(full_train_aug)))
    random.Random(SEED).shuffle(indices)

    train_size = int(len(indices) * TRAIN_SPLIT)
    val_size = int(len(indices) * VAL_SPLIT)

    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[train_size + val_size :]

    train_set = Subset(full_train_aug, train_indices)
    val_set = Subset(full_train_eval, val_indices)
    test_set = Subset(full_train_eval, test_indices)

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )
    test_loader = DataLoader(
        test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    print(
        f"\nWorking split sizes -> Train: {len(train_set)}, "
        f"Val: {len(val_set)}, Test: {len(test_set)}"
    )

    model = SimpleVehicleCNN(len(CLASS_NAMES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_model_state = copy.deepcopy(model.state_dict())
    best_val_accuracy = 0.0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        total_batches = len(train_loader)

        for batch_index, (images, labels, _) in enumerate(train_loader, start=1):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            predictions = outputs.argmax(dim=1)
            total_loss += loss.item() * images.size(0)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

            filled = int(20 * batch_index / total_batches)
            bar = "#" * filled + "-" * (20 - filled)
            print(
                f"\rEpoch {epoch + 1}/{EPOCHS} [{bar}] "
                f"{batch_index}/{total_batches}",
                end="",
            )

        print()

        train_loss = total_loss / total_samples
        train_accuracy = total_correct / total_samples
        val_loss, val_accuracy, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        if val_accuracy >= best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = copy.deepcopy(model.state_dict())

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy * 100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy * 100:.2f}% | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

    model.load_state_dict(best_model_state)
    test_loss, test_accuracy, test_labels, test_predictions = evaluate(
        model, test_loader, criterion, device
    )
    print(f"\nTest Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy * 100:.2f}%")

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, EPOCHS + 1), train_losses, label="Train Loss")
    plt.plot(range(1, EPOCHS + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "loss_curve.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, EPOCHS + 1), train_accuracies, label="Train Accuracy")
    plt.plot(range(1, EPOCHS + 1), val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "accuracy_curve.png")
    plt.close()

    confusion = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=int)
    for true_label, predicted_label in zip(test_labels, test_predictions):
        confusion[true_label, predicted_label] += 1

    plt.figure(figsize=(8, 6))
    plt.imshow(confusion, cmap="Blues")
    plt.colorbar()
    plt.xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45, ha="right")
    plt.yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    for row in range(confusion.shape[0]):
        for col in range(confusion.shape[1]):
            plt.text(
                col,
                row,
                str(confusion[row, col]),
                ha="center",
                va="center",
                color="black",
            )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png")
    plt.close()

    sample_count = min(LOCALIZATION_SAMPLES, len(test_set))
    sample_indices = random.Random(7).sample(range(len(test_set)), sample_count)
    cols = min(4, sample_count)
    rows = int(np.ceil(sample_count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)

    model.eval()
    for axis, sample_index in zip(axes, sample_indices):
        image_tensor, label, rel_path = test_set[sample_index]
        with torch.no_grad():
            logits, features = model(image_tensor.unsqueeze(0).to(device), return_features=True)

        predicted_label = int(logits.argmax(dim=1).item())
        class_weights = model.fc.weight[label].detach()
        cam = torch.einsum("c,chw->hw", class_weights, features[0])
        cam = torch.relu(cam)
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        cam = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=(image_tensor.shape[1], image_tensor.shape[2]),
            mode="bilinear",
            align_corners=False,
        ).squeeze().cpu().numpy()

        ys, xs = np.where(cam >= CAM_THRESHOLD)
        if len(xs) == 0 or len(ys) == 0:
            x1, y1 = 0, 0
            x2, y2 = image_tensor.shape[2] - 1, image_tensor.shape[1] - 1
        else:
            x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()

        image = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)
        image = image * np.array(STD) + np.array(MEAN)
        image = np.clip(image, 0.0, 1.0)

        axis.imshow(image)
        axis.imshow(cam, cmap="jet", alpha=0.28)
        axis.add_patch(
            plt.Rectangle(
                (x1, y1),
                max(1, x2 - x1),
                max(1, y2 - y1),
                fill=False,
                edgecolor="lime",
                linewidth=2,
            )
        )
        axis.set_title(
            f"T: {CLASS_NAMES[label]}\nP: {CLASS_NAMES[predicted_label]}",
            color="green" if predicted_label == label else "red",
        )
        axis.set_xlabel(Path(rel_path).name, fontsize=8)
        axis.axis("off")

    for axis in axes[sample_count:]:
        axis.axis("off")

    fig.suptitle("Weakly Supervised Localization with CAM", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "localized_predictions.png")
    plt.close(fig)

    torch.save(model.state_dict(), OUTPUT_DIR / "vehicle10_cnn.pth")
    print(f"\nSaved outputs to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
