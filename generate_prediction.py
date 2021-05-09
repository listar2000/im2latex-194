import torch
import argparse
from skimage.io import imread

from torch_utils import device
from build_vocab import load_vocab, Vocab, START_TOKEN, END_TOKEN
from models.encoder import Encoder
from models.row_encoder import RowEncoder
from models.decoder import DecoderWithAttention
from eval import beam_search
from utils import idx2formulas


class LatexGenerator(object):

    def __init__(self, model_path, beam_size=5):
        self.model_path = model_path
        self.beam_size = beam_size

        # Load vocab
        self.vocab = load_vocab()
        self.vocab_size = len(self.vocab)

        # Load model
        self.encoder, self.row_encoder, self.decoder = self.load_model(model_path)
        

    def load_model(self, model_path):
        checkpoint = torch.load(model_path)
        encoder = Encoder()
        encoder.load_state_dict(checkpoint['encoder'])
        encoder.to(device)
        encoder.eval()
        if "row_encoder" in checkpoint:
            row_encoder = RowEncoder(train_init=False)

            row_state_dict = checkpoint['row_encoder']
            for k, v in row_state_dict.items():
                if k == "init_hidden":
                    row_encoder.init_hidden = v
                elif k == "init_cell":
                    row_encoder.init_cell = v
            row_encoder.load_state_dict(row_state_dict, strict=False)

            row_encoder.to(device)
            row_encoder.eval()
        else:
            row_encoder = None
        decoder = DecoderWithAttention(vocab_size=self.vocab_size)
        decoder.load_state_dict(checkpoint['decoder'])
        decoder.to(device)
        decoder.eval()
        return encoder, row_encoder, decoder


    def generate_prediction(self, img_path):
        # Load and preprocess image
        img = imread(img_path)
        img = img.transpose(2, 0, 1)
        img = torch.FloatTensor(img / 255.)
        img = img.unsqueeze(0) # dimension for batch

        # Predict
        img = img.to(device)
        encoded_img = self.encoder(img)
        if self.row_encoder is not None:
            encoded_img = self.row_encoder(encoded_img)
        pred_ind, alphas = beam_search(encoded_img, self.encoder, self.row_encoder, self.decoder, self.vocab_size, self.beam_size)
        predictions = idx2formulas([pred_ind[1:]], self.vocab) # (1, pred_length)
        pred_str = "".join(predictions[0])

        return pred_str



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predicting...")
    parser.add_argument("--img_path", type=str, help="Path to image to predict")
    parser.add_argument("--model_path", type=str, default="BEST.pth.tar", help="Which checkpoint to use")
    parser.add_argument('--beam_size', type=int, default=5, help='Beam size for beam search')
    args = parser.parse_args()

    latex_generator = LatexGenerator(args.model_path, args.beam_size)
    pred_str = latex_generator.generate_prediction(args.img_path)
    print(pred_str)


