import torch

resnet18_config = {
    'modelname': 'ResNet18',
    'modeltype': 'ResidualNetwork',
    'parameters': dict(out_dimensions=3, model_dtype=torch.float64, model_device="cuda:0")
}
