import torch
import numpy as np
from config import train_config

device = train_config["device"]

def init_gpu(use_gpu=train_config['use_gpu'], gpu_id=train_config['gpu_id']):
    global device
    if torch.cuda.is_available() and use_gpu:
        device = torch.device("cuda:" + str(gpu_id))
        print("Using GPU id {}".format(gpu_id))
    else:
        device = torch.device("cpu")
        print("GPU not detected. Defaulting to CPU.")

def set_device(gpu_id):
    torch.cuda.set_device(gpu_id)

def from_numpy(nparray, dtype: torch.dtype = None):
    # let torch do the automatic type inference
    if not dtype: 
        return torch.from_numpy(nparray).to(device)
    else:
        return torch.from_numpy(nparray).type(dtype).to(device)

def to_numpy(tensor, dtype: np.dtype = None):
    if not dtype:
        return tensor.to('cpu').detach().numpy()
    else:
        return tensor.to('cpu').detach().numpy().astype(dtype)

def optimizer_to_device(optimizer, device):
    # Adapted from https://github.com/pytorch/pytorch/issues/2830
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)