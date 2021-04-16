from typing import Dict
import torch
from torch import nn
import torchvision
# configuration file
from config import train_config

backbone_map = {
    'ResNet101': torchvision.models.resnet101,
    'AlexNet': torchvision.models.alexnet
}

class Encoder(nn.Module):
    
    def __init__(self, 
                 backbone=train_config['cnn_backbone'],
                 encoded_img_size=train_config['encoded_img_size']
                ):
        super(Encoder, self).__init__()

        if backbone not in backbone_map:
            backbone = 'ResNet101' # use resnet101 as default

        backbone_net: nn.Module = backbone_map[backbone](pretrained=True)
        
        # the deletion of adaptive pooling & fc layers is model dependent
        if backbone in ['ResNet101', 'AlexNet']:
            modules = list(backbone_net.children())[:-2]
            self.backbone_net = nn.Sequential(*modules)
        
        self.backbone: str = backbone
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_img_size, encoded_img_size))
        self.fine_tune()
    
    def forward(self, images):
        """
        Forward propagation.

        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images

        out_channels:
        resnet101: 2048
        alexnet: 256

        out_img_size:
        resnet101: image_size/32 
        alexnet: hard to directly compute
        """
        out = self.backbone_net(images)  # (batch_size, out_channels, out_img_height, out_img_width)
        # out = self.adaptive_pool(out)  # (batch_size, out_channels, encoded_image_height, encoded_image_width)
        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_height, encoded_image_width, out_channels)
        return out
    
    def fine_tune(self, fine_tune=True):
        for p in self.backbone_net.parameters():
            p.requires_grad = False

        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        if self.backbone == 'ResNet101':
            for c in list(self.backbone_net.children())[5:]:
                for p in c.parameters():
                    p.requires_grad = fine_tune
        # only fine-tune the last conv2d + relu + maxpool modules
        elif self.backbone == 'AlexNet':
            for layer in self.backbone_net.features[10:]:
                for p in layer.parameters():
                    p.requires_grad = fine_tune


    