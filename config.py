import torch
import os

# configuration params for training the neural network

# data path related configs
DATA_FOLDER_PATH = './data'
IMG_FOLDER_PATH = os.path.join(DATA_FOLDER_PATH, 'images_processed') # 'formula_images_processed' for full data
PROCESSED_FOLDER_PATH = './processed_data'

preprocess_config = {
    'min_freq': 5, # minimum frequency for a token not to be considered unknown
    'max_len': 150, # maximum length of a caption to be taken in the training set
    'padding': False, # whether or not to pad the 

    # parameters for resizing & standardizing the images
    'resize': True,
    'resize_width': 256,
    'resize_height': 256
}

train_config = {
    'use_cuda': False,
    'num_workers': 4,
    'lr': 3e-4,
    'max_epoch': 10,
    # 1) encoder (CNN) related configs
    # see `backbone_map` in encoder.py for a list of supported CNN backbones
    'cnn_backbone': 'ResNet101', # {'ResNet101', 'AlexNet'}
    'encoded_img_size': 14,
    'batch_size': 20,

    # 2) attention related configs
    'attention_dim': 2048,
    'alpha_c': 1, # regularization parameter for 'doubly stochastic attention'

    # 3) decoder (LSTM) related configs
    'decoder_dim': 512
}