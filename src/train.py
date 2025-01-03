import os
from glob import glob
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score, precision_recall_curve, roc_curve

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

class TrainDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.transform = transform

        self.data_files = []
        self.labels = []

        fake_dir = os.path.join(data_dir, "fake_sdv1/inversion") 
        real_dir = os.path.join(data_dir, "real_laion/inversion") 

        for file in glob(os.path.join(fake_dir, "*.pt")):
            self.data_files.append(file)
            self.labels.append(0) # label: 0

        for file in glob(os.path.join(real_dir, "*.pt")):
            self.data_files.append(file)
            self.labels.append(1)  # label: 1

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        data_path = self.data_files[idx]
        data = torch.load(data_path).squeeze(0) 
        label = self.labels[idx]
        if self.transform:
            data = self.transform(data)
        return data, label

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/data/yskim/pix2pix-zero/output')
    parser.add_argument('--model_save_path', type=str, default='/data/yskim/pix2pix-zero/fake_sdv1-real_laion.pth')
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()

    # Dataset and DataLoader
    transform = transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
        transforms.Normalize(mean=[0.5] * 9, std=[0.5] * 9)
    ])
    dataset = TrainDataset(args.data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Load ResNet-50
    model = models.resnet50(pretrained=True)
    model.conv1 = nn.Conv2d(9, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        y_true, y_scores = [], []
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device, dtype=torch.float32)

            optimizer.zero_grad()
            outputs = model(inputs).squeeze()

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            y_true.extend(labels.cpu().tolist())
            y_scores.extend(torch.sigmoid(outputs).detach().cpu().tolist())

        aucroc = roc_auc_score(y_true, y_scores)
        avg_precision = average_precision_score(y_true, y_scores)
        accuracy = accuracy_score(y_true, [1 if s > 0.5 else 0 for s in y_scores])

        print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {running_loss / len(dataloader):.4f}, AUC-ROC: {aucroc:.4f}, Avg Precision: {avg_precision:.4f}, Accuracy: {accuracy:.4f}")

    print("Training complete.")
    
    torch.save(model.state_dict(), args.model_save_path)
    print(f"Model saved to {args.model_save_path}")