import numpy as np
import torch
from torch import nn
# configuration file
from config import train_config
from torch_utils import from_numpy, device

class RowEncoder(nn.Module):
        
    def __init__(self, input_size=train_config['row_input_size'], 
                       hidden_size=train_config["row_hidden_size"],
                       train_init=train_config["row_init"]):
        """
        if train_init is TRUE, then the initial 
        """

        super(RowEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        # flag for whether the initialization network has been created
        self.train_init = train_init
        self.init_created = False 
    
    def forward(self, images):
        """
        Forward propagation.

        Input:
            > images: images, a tensor of dimensions (batch_size, image_size, image_size, num_channels)
        Output:
            > encoder_outputs: encoded images


        """
        b, h, w, c = images.shape

        if self.train_init and not self.init_created:
            self.init_hidden = torch.empty(2, h, self.hidden_size)
            self.init_hidden = nn.Parameter(nn.init.kaiming_normal_(self.init_hidden).to(device))
            self.init_cell = torch.empty(2, h, self.hidden_size)
            self.init_cell = nn.Parameter(nn.init.kaiming_normal_(self.init_cell).to(device))
            self.init_created = True

        images = images.contiguous().view(b*h, w, c)

        init_hc = (self.init_hidden.repeat(1, b, 1), self.init_cell.repeat(1, b, 1))
        # init_hc = (torch.zeros(2, b*h, self.hidden_size).to(device), torch.zeros(2, b*h, self.hidden_size).to(device))
        encoded_outputs, (hn, cn) = self.lstm(images, init_hc)
        # encoded_outputs -> (b * h, w, 2 * hidden_size)
        encoded_outputs = encoded_outputs.view(b, h, w, -1).contiguous()
        encoded_outputs = encoded_outputs.view(b, h*w, -1)
        # encoded_outputs = self.add_pos_embedding(encoded_outputs)
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

    
    def hidden_init_network(self, row_num):
        network = []
        network.append(nn.Linear(row_num, 512))
        network.append(nn.ReLU())
        network.append(nn.Linear(512, 512))
        network.append(nn.ReLU())
        network.append(nn.Linear(512, row_num))

        return nn.Sequential(*network)

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


    