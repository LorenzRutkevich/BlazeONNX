import torch
import torchvision.models as models
from blazeface import BlazeFace

model_path = "blazeface.pth"
model = BlazeFace()
model.load_weights(model_path)

# Export the model to ONNX
dummy_input = torch.randn(1, 3, 128, 128)
torch.onnx.export(model, dummy_input, "blazeface.onnx", verbose=True)
