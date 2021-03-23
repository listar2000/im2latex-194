import torch

# configuration params for training the neural network
train_config = {

    # 1) file paths related configs
    'file_path': 'foo_bar_bee',

    # 2) encoder (CNN) related configs
    # see `backbone_map` in encoder.py for a list of supported CNN backbones
    'cnn_backbone': 'ResNet101',
    'encoded_img_size': 14,

    # 3) attention related configs
    'attention_dim': 2048,

    # 4) decoder (LSTM) related configs
    'decoder_dim': 2048
}