import torch
import numpy as np
from config import train_config

device = None

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

def from_numpy(dtype: torch.dtype = None, *args, **kwargs):
    # let torch do the automatic type inference
    if not dtype: 
        return torch.from_numpy(*args, **kwargs).to(device)
    else:
        return torch.from_numpy(*args, **kwargs).type(dtype).to(device)

def to_numpy(tensor, dtype: np.dtype = None):
    if not dtype:
        return tensor.to('cpu').detach().numpy()
    else:
        return tensor.to('cpu').detach().numpy().astype(dtype)
