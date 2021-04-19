import torch
from os.path import join
import torch.nn.functional as F
import torch.optim as optim
import argparse
from tqdm import tqdm
import wandb
from nltk.translate.bleu_score import sentence_bleu

from torch_utils import to_numpy, device
from dataset import LatexDataloader
from build_vocab import load_vocab, Vocab, START_TOKEN, END_TOKEN
from models.encoder import Encoder
from models.row_encoder import RowEncoder
from models.decoder import DecoderWithAttention
from config import test_config, train_config
from utils import *

def load_model(vocab_size, model_name, sample, use_row):
    if sample:
        model_folder = "checkpoint/sample"
    else:
        model_folder = "checkpoint"
    model_path = join(model_folder, model_name)
    checkpoint = torch.load(model_path)

    encoder = Encoder()
    encoder.load_state_dict(checkpoint['encoder'])
    encoder.to(device)
    encoder.eval()
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=train_config["lr"])
    encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer'])

    if use_row:
        row_encoder = RowEncoder()
        row_encoder.load_state_dict(checkpoint['row_encoder'])
        row_encoder.to(device)
        row_encoder.eval()
        row_encoder_optimizer = optim.Adam(row_encoder.parameters(), lr=train_config["lr"])
        row_encoder_optimizer.load_state_dict(checkpoint['row_encoder_optimizer'])

    else:
        row_encoder = None
        row_encoder_optimizer = None

    decoder = DecoderWithAttention(vocab_size=vocab_size)
    decoder.load_state_dict(checkpoint['decoder'])
    decoder.to(device)
    decoder.eval() 
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=train_config["lr"])
    decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer'])

    return encoder, row_encoder, decoder, encoder_optimizer, row_encoder_optimizer, decoder_optimizer


def test(test_loader, encoder, row_encoder, decoder, vocab, beam_size):
    bleu = AverageMeter()
    for i, (img, form, form_len) in enumerate(tqdm(test_loader, unit="test_batch")):
        img = img.to(device)
        form = form.to(device)
        form_len = form_len.to(device)

        encoded_img = encoder(img)
        if row_encoder is not None:
            encoded_img = row_encoder(encoded_img)

        encoder_dim = encoded_img.size(-1)
        # Flatten image
        encoded_img = encoded_img.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
        num_pixels = encoded_img.size(1)


        # ============= Decoding by beam search ===============
        k = beam_size

        # We'll treat the problem as having a batch size of k
        encoded_img = encoded_img.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

        # Tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.LongTensor([START_TOKEN] * k).to(device)  # (k)

        # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words.unsqueeze(1)  # (k, 1)

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k).to(device)  # (k)
        
        h, c = decoder.init_hidden_state(encoded_img)  # (k, decoder_dim)

        # Lists to store completed sequences and scores
        complete_seqs = list()
        complete_seqs_avg_scores = list()

        step = 1
        while True:
            embeddings = decoder.embedding(k_prev_words.unsqueeze(1)).squeeze(1)  # (k, embed_dim)
            attention_weighted_encoding, _ = decoder.attention(encoded_img, h)  # (k, encoder_dim), (k, num_pixels)
            gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (k, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = decoder.decode_step(torch.cat([embeddings, attention_weighted_encoding], dim=1),
                                       (h, c))  # (k, decoder_dim)
            scores = decoder.fc(h)  # (k, vocab_size), drop_out is removed during inference
            scores = F.log_softmax(scores, dim=1)

            scores_so_far = top_k_scores.unsqueeze(1).expand_as(scores) + scores # (k, vocab_size)

            # Unroll and find top scores, and their unrolled indices
            top_k_scores, top_k_words_inds = scores_so_far.view(-1).topk(k, dim=0)  # (k), (k)

            # Convert unrolled indices to actual indices of scores
            prev_word_inds = top_k_words_inds // vocab_size # (k)
            new_word_inds = top_k_words_inds % vocab_size # (k)

            # Add new words to sequences
            seqs = torch.cat([seqs[prev_word_inds], new_word_inds.unsqueeze(1)], dim=1)  # (k, step+1)

            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [i for i, word in enumerate(new_word_inds) if word != END_TOKEN]
            complete_inds = [i for i, word in enumerate(new_word_inds) if word == END_TOKEN]

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_avg_scores.extend(top_k_scores[complete_inds]/(step+1))
            k -= len(complete_inds)  # reduce beam length accordingly

            # Break if we have all k candidates or if things have been going for too long
            if k == 0 or step > test_config["max_length"]: 
                if len(complete_inds) == 0:
                    complete_seqs.extend(seqs.tolist())
                    complete_seqs_avg_scores.extend(top_k_scores/(step+1))
                break

            # Proceed with incomplete sequences
            h = h[prev_word_inds[incomplete_inds]] # (k, decoder_dim)
            c = c[prev_word_inds[incomplete_inds]] # (k, decoder_dim)
            encoded_img = encoded_img[prev_word_inds[incomplete_inds]] # (k, num_pixels, encoder_dim)
            seqs = seqs[incomplete_inds] # (k, step+1)
            top_k_scores = top_k_scores[incomplete_inds] # (k)
            k_prev_words = new_word_inds[incomplete_inds] # (k)

            step += 1


        # Calculate max score/word
        max_i = complete_seqs_avg_scores.index(max(complete_seqs_avg_scores))
        pred_ind = complete_seqs[max_i]
        predictions = idx2formulas([pred_ind], vocab) # (1, pred_length)
        references = idx2formulas(to_numpy(form), vocab) # (1, ref_length)

        assert(len(predictions) == len(references) == 1)

        # Sample prediction in wandb
        pred_str = "".join(predictions[0])
        wandb.log({"test_pred_examples": [wandb.Image(img[0], caption=pred_str)]})

        # Calculate BLEU score
        for ref, pred in zip(references, predictions):
            bleu_score = sentence_bleu([ref], pred)
        bleu.update(bleu_score)

        # print every 10 batches
        if i % 10 == 0:
            print('Test: [{0}/{1}]\t'
                  'BLEU {2} ({bleu.avg:.3f})'.format(i, len(test_loader), bleu_score, bleu=bleu))

    return bleu.avg



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training...")
    parser.add_argument("--sample", action="store_true", default=False, help="Use sample data or not")
    parser.add_argument("--model_name", type=str, default="BEST.pth.tar", help="Which checkpoint to use")
    parser.add_argument("--row", action="store_true", default=False, help="Use row_encoder or not")
    parser.add_argument('--beam_size', type=int, default=5, help='Beam size for beam search')
    args = parser.parse_args()

    vocab = load_vocab()
    vocab_size = len(vocab)

    print("Loading test data...")
    test_loader = LatexDataloader("test", batch_size=1, shuffle=True, sample=args.sample)

    print("Loading model...")
    encoder, row_encoder, decoder, encoder_optimizer, row_encoder_optimizer, decoder_optimizer = load_model(vocab_size, args.model_name, args.sample, args.row)

    wandb.init(project="im2latex")
    avg_bleu = test(test_loader=test_loader,
                    encoder=encoder,
                    row_encoder=row_encoder,
                    decoder=decoder, 
                    vocab = vocab,
                    beam_size = args.beam_size)

