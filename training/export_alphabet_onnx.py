"""Export the production MobileNet alphabet checkpoint to ONNX."""
from pathlib import Path
import torch
from torchvision.models import mobilenet_v3_large
checkpoint=torch.load('backend/checkpoints/alphabet_mobilenetv3.pt',map_location='cpu',weights_only=False)
model=mobilenet_v3_large(weights=None); model.classifier[3]=torch.nn.Linear(model.classifier[3].in_features,len(checkpoint['labels'])); model.load_state_dict(checkpoint['model_state_dict']); model.eval()
Path('backend/checkpoints').mkdir(exist_ok=True)
torch.onnx.export(model,torch.randn(1,3,224,224),'backend/checkpoints/alphabet_mobilenetv3.onnx',input_names=['image'],output_names=['logits'],dynamic_axes={'image':{0:'batch'},'logits':{0:'batch'}},opset_version=17)
print('Exported backend/checkpoints/alphabet_mobilenetv3.onnx')
