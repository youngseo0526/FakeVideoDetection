import os
from glob import glob
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve, auc
from torch_fidelity import calculate_metrics

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

class EvalDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.data_files = sorted(glob(os.path.join(self.data_dir, "*.pt")))  

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        data_path = self.data_files[idx]
        data = torch.load(data_path).squeeze(0) 
        label = 1 if "real" in data_path else 0  # Assign labels based on file path
        if self.transform:
            data = self.transform(data)
        return data, label

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/data/yskim/pix2pix-zero/output')
    parser.add_argument('--model_path', type=str, default="/data/yskim/pix2pix-zero/fake_sdv1-real_laion.pth")
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    # Dataset and DataLoader
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.5] * 9, std=[0.5] * 9)
    ])
    dataset = EvalDataset(args.data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Load ResNet-50
    model = models.resnet50()
    model.conv1 = torch.nn.Conv2d(9, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(torch.load(args.model_path))  # Load trained model
    model = model.to(device)
    model.eval()

    # Metrics
    criterion = nn.BCELoss()
    total_loss = 0.0
    all_labels = []
    all_probs = []

    # Evaluation loop
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device, dtype=torch.float32)
            outputs = model(inputs).squeeze()
            probs = torch.sigmoid(outputs)  # Apply sigmoid to get probabilities

            all_probs.extend(probs.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    # Compute AUC-ROC, Accuracy, and PR-AUC
    auc_roc = roc_auc_score(all_labels, all_probs)
    accuracy = accuracy_score(all_labels, [1 if p > 0.5 else 0 for p in all_probs])
    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
    pr_auc = auc(recall, precision)

    print(f"AUC-ROC: {auc_roc:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")

    # Compute FID and KID using torch-fidelity
    metrics = calculate_metrics(
        input1=args.real_dir,
        input2=args.fake_dir,
        cuda=device == "cuda",
        isc=False,  # Skip Inception Score
        fid=True,   # Calculate FID
        kid=True    # Calculate KID
    )

    print(f"FID: {metrics['frechet_inception_distance']:.4f}")
    print(f"KID: {metrics['kernel_inception_distance_mean']:.4f}")