import torch
import os

# configuration params for training the neural network

# data path related configs
DATA_FOLDER_PATH = 'H:/CS194-project/data'
IMG_FOLDER_PATH = os.path.join(DATA_FOLDER_PATH, 'formula_images_processed')
PROCESSED_FOLDER_PATH = 'H:/CS194-project/im2latex-194/processed_data'

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
    # automatically use the GPU if there's one available
    'use_gpu': True,
    'gpu_id': 0,
    
    # 1) encoder (CNN) related configs
    # see `backbone_map` in encoder.py for a list of supported CNN backbones
    'cnn_backbone': 'ResNet101',
    'encoded_img_size': 14,

    # 2) attention related configs
    'attention_dim': 2048,

    # 3) decoder (LSTM) related configs
    'decoder_dim': 2048
}