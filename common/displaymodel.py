from torch import nn
import torch

def display_model(model):
    print(model)
    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params}")