import torch
from torchvision.models import mobilenet_v3_small

model = mobilenet_v3_small(pretrained=True)
torch.save(model.state_dict(), "mobilenet_v3_small.pth")
