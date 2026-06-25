from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from model import build_model


class EuroSATDataset(Dataset):
    def __init__(self, dataframe, label_to_idx, project_root, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.label_to_idx = label_to_idx
        self.project_root = Path(project_root)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image_path = self.project_root / row["image_path"].replace("..\\", "").replace("../", "")
        label = self.label_to_idx[row["label"]]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def plot_and_save_confusion_matrix(cm, class_names, save_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=range(len(class_names)),
        yticks=range(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted label",
        ylabel="True label",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    project_root = Path(__file__).resolve().parent.parent
    csv_path = project_root / "data" / "processed" / "dataset_index.csv"
    model_path = project_root / "models" / "best_model.pt"
    logs_dir = project_root / "experiments" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    checkpoint = torch.load(model_path, map_location="cpu")
    class_names = checkpoint["class_names"]
    label_to_idx = checkpoint["label_to_idx"]
    num_classes = len(class_names)

    _, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    val_dataset = EuroSATDataset(val_df, label_to_idx, project_root, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    metrics = {
        "accuracy": acc,
        "precision_weighted": precision,
        "recall_weighted": recall,
    }

    metrics_path = logs_dir / "metrics.csv"
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

    cm_path = logs_dir / "confusion_matrix.png"
    plot_and_save_confusion_matrix(cm, class_names, cm_path)

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved confusion matrix to: {cm_path}")


if __name__ == "__main__":
    main()
