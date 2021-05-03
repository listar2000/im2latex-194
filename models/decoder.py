"""
Adapted from 
https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning/blob/master/models.py

"""

import torch
from torch import nn
import torchvision
from models.attention import Attention

class DecoderWithAttention(nn.Module):
    """
    Decoder.
    """

    def __init__(self, vocab_size, attention_dim=512, embed_dim=512, decoder_dim=512, encoder_dim=2048, dropout=0.5):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(DecoderWithAttention, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # attention network

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        """
        Loads embedding layer with pre-trained embeddings.
        :param embeddings: pre-trained embeddings
        """
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).
        :param fine_tune: Allow?
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, images):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.
        :param images: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_images = images.mean(dim=1)
        h = self.init_h(mean_images)  # (batch_size, decoder_dim)
        c = self.init_c(mean_images)
        return h, c

    def forward(self, images, formulas, formula_lengths, epsilon=1.):
        """
        Forward propagation.

        Input:
            > images: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
            > formulas: encoded formulas, a tensor of dimension (batch_size, max_formula_length)
            > formula_lengths: formula lengths, a tensor of dimension (batch_size)
        Output:
            > scores: scores for vocabulary -> (batch_size, max_formula_length, vocab_size)
            > alphas: weights -> (batch_size, max_formula_length, num_pixels)
        """

        batch_size = images.size(0)
        encoder_dim = images.size(-1)
        vocab_size = self.vocab_size

        # Flatten image
        images = images.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = images.size(1)

        # (Commented out - Already sorted in dataloader)
        # # Sort input data by decreasing lengths; why? apparent below
        # formula_lengths, sort_ind = formula_lengths.squeeze(1).sort(dim=0, descending=True)
        # images = images[sort_ind]
        # formulas = formulas[sort_ind]

        # Embedding
        embeddings = self.embedding(formulas)  # (batch_size, max_formula_length, embed_dim)

        # Initialize LSTM state
        h, c = self.init_hidden_state(images)  # (batch_size, decoder_dim)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (formula_lengths + 1).tolist() # include start, not end tokens

        # Initialize tensors to hold word prediction scores and alphas
        scores = torch.zeros(batch_size, max(decode_lengths), vocab_size)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels)

        # At each time-step, decode by:
        # 1) attention-weighing the encoder's output based on the decoder's previous hidden state output
        # 2) then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths]) # number of batches to decode
            attention_weighted_encoding, alpha = self.attention(images[:batch_size_t], h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
            # randomly not following the teacher forcing
            if t > 1 and torch.rand(1) > epsilon:
                h, c = self.decode_step(torch.cat([scores[:batch_size_t, t-1, :], 
                                               attention_weighted_encoding], dim=1),
                                    (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
            else:
                h, c = self.decode_step(torch.cat([embeddings[:batch_size_t, t, :], 
                                                attention_weighted_encoding], dim=1),
                                        (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)

            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            scores[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        # # Remove the <start> token
        # scores = scores[:, 1:, :]
        # alphas = alphas[:, 1:, :]
        return scores, alphas

