import numpy as np
import torch
from torch import nn
# configuration file
from config import train_config
from torch_utils import from_numpy

device = train_config["device"]

class RowEncoder(nn.Module):
    
    def __init__(self, input_size=2048, hidden_size=train_config["row_hidden_size"]):
        super(RowEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
    
    def forward(self, images):
        """
        Forward propagation.

        Input:
            > images: images, a tensor of dimensions (batch_size, image_size, image_size, num_channels)
        Output:
            > encoder_outputs: encoded images


        """
        b, h, w, c = images.shape

        images = images.contiguous().view(b*h, w, c)
        init_hidden = (torch.zeros(2, b*h, self.hidden_size).to(device), torch.zeros(2, b*h, self.hidden_size).to(device))
        encoded_outputs, (hn, cn) = self.lstm(images, init_hidden)
        # encoded_outputs -> (b * h, w, 2 * hidden_size)
        encoded_outputs = encoded_outputs.view(b, h, w, -1).contiguous()
        encoded_outputs = encoded_outputs.view(b, h*w, -1)
        encoded_outputs = self.add_pos_embedding(encoded_outputs)
        encoded_outputs = encoded_outputs.view(b, h, w, -1).contiguous()

        return encoded_outputs


    def add_pos_embedding(self, x):
        """
        Input:
            > x: (batch, height * width, directions * hidden_size)
        Output:
            > x + positional_embedding
        """
        n_position = x.shape[1]
        emb_dim = x.shape[2]
        position_enc = np.array([
                        [pos / np.power(10000, 2 * (j // 2) / emb_dim) for j in range(emb_dim)]
                         for pos in range(n_position)])
        position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])
        position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])
        return x + from_numpy(position_enc, dtype=torch.FloatTensor)










# class RowEncoder(nn.Module):
    
#     def __init__(self, input_size=2048, hidden_size=1024):
#         super(RowEncoder, self).__init__()
#         self.hidden_size = hidden_size
#         self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
    
#     def forward(self, images):
#         """
#         Forward propagation.

#         :param images: images, a tensor of dimensions (batch_size, num_channels, image_size, image_size)
#         :return: encoded images

#         """
#         b, h, w, c = images.shape

#         # Add positional embedding to each row of an image
#         pos_embedding = torch.arange(h).expand(b, c, 1, h).permute(0, 3, 2, 1)
#         embedded_imgs = torch.cat((pos_embedding, images), dim=2)
#         out_imgs = torch.zeros(b, h, w+1, self.hidden_size*2)
#         for row in range(h):
#             row_out, (hn, cn) = self.lstm(embedded_imgs[:, row, :, :]) # row_out: (batch, seq_len, num_dir*hidden_size)
#             out_imgs[:, row, :, :] = row_out
#         return out_imgs


    