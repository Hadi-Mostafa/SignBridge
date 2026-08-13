"""Evaluate a trained MobileNet ASL alphabet checkpoint on a held-out test split."""
import argparse
import time
from pathlib import Path
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v3_large

parser=argparse.ArgumentParser(); parser.add_argument('--data',type=Path,required=True); parser.add_argument('--checkpoint',type=Path,default=Path('backend/checkpoints/alphabet_mobilenetv3.pt')); parser.add_argument('--batch-size',type=int,default=32); args=parser.parse_args()
checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False); labels=checkpoint['labels']
model=mobilenet_v3_large(weights=None); model.classifier[3]=torch.nn.Linear(model.classifier[3].in_features,len(labels)); model.load_state_dict(checkpoint['model_state_dict']); model.eval()
transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
dataset=datasets.ImageFolder(args.data/'test',transform); loader=DataLoader(dataset,args.batch_size)
truth=[]; predicted=[]; started=time.perf_counter()
with torch.inference_mode():
 for images,target in loader:
  predicted.extend(model(images).argmax(1).tolist()); truth.extend(target.tolist())
elapsed=time.perf_counter()-started
print(classification_report(truth,predicted,target_names=labels,digits=4)); print('confusion_matrix='); print(confusion_matrix(truth,predicted)); print(f'fps={len(dataset)/elapsed:.2f}')
