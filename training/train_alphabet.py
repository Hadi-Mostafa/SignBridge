"""Fine-tune MobileNetV3-Large on a real, camera-style ASL alphabet dataset.

Expected input layout (do not use Sign Language MNIST):
data/asl_alphabet/{train,val,test}/A/*.jpg ... /Z/*.jpg
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large


def transforms_for(train: bool):
    operations = [transforms.Resize((224, 224))]
    if train:
        operations += [
            transforms.RandomRotation(12), transforms.ColorJitter(brightness=.25, contrast=.2, saturation=.15),
            transforms.RandomResizedCrop(224, scale=(.75, 1.0)),
            # Do not flip by default: several ASL letters are handed/asymmetric.
        ]
    operations += [transforms.ToTensor(), transforms.Normalize([.485,.456,.406], [.229,.224,.225])]
    return transforms.Compose(operations)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--output", type=Path, default=Path("backend/checkpoints/alphabet_mobilenetv3.pt"))
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set = datasets.ImageFolder(args.data / "train", transforms_for(True))
    val_set = datasets.ImageFolder(args.data / "val", transforms_for(False))
    if train_set.classes != val_set.classes: raise ValueError("train and val labels must match")
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, num_workers=2, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_set, args.batch_size, num_workers=2, pin_memory=device.type == "cuda")
    model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(train_set.classes))
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    criterion, best, stale = nn.CrossEntropyLoss(), 0.0, 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = criterion(model(images.to(device)), labels.to(device))
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        model.eval(); correct = total = 0
        with torch.inference_mode():
            for images, labels in val_loader:
                logits = model(images.to(device)); correct += (logits.argmax(1).cpu() == labels).sum().item(); total += len(labels)
        accuracy = correct / max(total, 1)
        print(f"epoch={epoch} val_accuracy={accuracy:.4f}")
        if accuracy > best:
            best, stale = accuracy, 0
            torch.save({"architecture":"mobilenet_v3_large", "labels":train_set.classes, "model_state_dict":model.state_dict(), "val_accuracy":accuracy}, args.output)
        else:
            stale += 1
            if stale >= args.patience: break
    print(json.dumps({"checkpoint":str(args.output), "best_val_accuracy":best, "device":str(device)}))

if __name__ == "__main__": main()
