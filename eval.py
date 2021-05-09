import torch
from os.path import join
import torch.nn.functional as F
import argparse
from tqdm import tqdm
import wandb
from nltk.translate.bleu_score import sentence_bleu
import torchvision.transforms as transforms
from PIL import Image

from torch_utils import to_numpy, device
from dataset import LatexDataloader
from build_vocab import load_vocab, Vocab, START_TOKEN, END_TOKEN
from models.encoder import Encoder
from models.row_encoder import RowEncoder
from models.decoder import DecoderWithAttention
from config import test_config, train_config
from utils import *

def load_model(model_folder, vocab_size, model_name, sample=False):
    if sample:
        model_folder = model_folder+"/sample"
    model_path = join(model_folder, model_name)
    checkpoint = torch.load(model_path)
    print("Checkpoint ended in epoch {}".format(checkpoint['epoch']))

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

    decoder = DecoderWithAttention(vocab_size=vocab_size)
    decoder.load_state_dict(checkpoint['decoder'])
    decoder.to(device)
    decoder.eval()

    return encoder, row_encoder, decoder

def beam_search(encoded_img, encoder, row_encoder, decoder, vocab_size, beam_size):
    
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

    # Tensor to store top k sequences' alphas; now they're just 1s
    seqs_alpha = torch.ones(k, 1, num_pixels).to(device)  # (k, 1, num_pixels)
    
    h, c = decoder.init_hidden_state(encoded_img)  # (k, decoder_dim)

    # Lists to store completed sequences and scores
    complete_seqs = list()
    complete_seqs_avg_scores = list()
    complete_seqs_alpha = list()

    step = 1
    while True:
        embeddings = decoder.embedding(k_prev_words.unsqueeze(1)).squeeze(1)  # (k, embed_dim)
        attention_weighted_encoding, alpha = decoder.attention(encoded_img, h)  # (k, encoder_dim), (k, num_pixels)
        gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (k, encoder_dim)
        attention_weighted_encoding = gate * attention_weighted_encoding
        h, c = decoder.decode_step(torch.cat([embeddings, attention_weighted_encoding], dim=1),
                                   (h, c))  # (k, decoder_dim)
        scores = decoder.fc(h)  # (k, vocab_size), drop_out is removed during inference
        scores = F.log_softmax(scores, dim=1)


        scores_so_far = top_k_scores.unsqueeze(1).expand_as(scores) + scores # (k, vocab_size)

        if step == 1:
            # For the first step, all k points will have the same scores, 
            # so we take just one row and find its top k scores.
            top_k_scores, top_k_words_inds = scores_so_far[0].topk(k, dim=0)  # (k), (k)
        else:
            # Unroll and find top scores, and their unrolled indices
            top_k_scores, top_k_words_inds = scores_so_far.view(-1).topk(k, dim=0)  # (k), (k)
            
        # Convert unrolled indices to actual indices of scores
        prev_word_inds = top_k_words_inds // vocab_size # (k)
        new_word_inds = top_k_words_inds % vocab_size # (k)

        # Add new words, alphas to sequences
        seqs = torch.cat([seqs[prev_word_inds], new_word_inds.unsqueeze(1)], dim=1)  # (k, step+1)
        seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)], dim=1)  # (k, step+1, num_pixels)

        # Which sequences are incomplete (didn't reach <end>)?
        incomplete_inds = [i for i, word in enumerate(new_word_inds) if word != END_TOKEN]
        complete_inds = [i for i, word in enumerate(new_word_inds) if word == END_TOKEN]

        # Set aside complete sequences
        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_avg_scores.extend(top_k_scores[complete_inds]/(step+1))
            complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
        # k -= len(complete_inds)  # reduce beam length accordingly

        # Break if we have all k candidates or if things have been going for too long
        if (len(complete_seqs) >= 10) or (step > test_config["max_length"]): 
            if step > test_config["max_length"]:
                complete_seqs.extend(seqs.tolist())
                complete_seqs_avg_scores.extend(top_k_scores/(step+1))
                complete_seqs_alpha.extend(seqs_alpha.tolist())
            break

        # Proceed with incomplete sequences
        h = h[prev_word_inds[incomplete_inds]] # (k, decoder_dim)
        c = c[prev_word_inds[incomplete_inds]] # (k, decoder_dim)
        encoded_img = encoded_img[prev_word_inds[incomplete_inds]] # (k, num_pixels, encoder_dim)
        seqs = seqs[incomplete_inds] # (k, step+1)
        seqs_alpha = seqs_alpha[incomplete_inds] # (k, step+1, num_pixels)
        top_k_scores = top_k_scores[incomplete_inds] # (k)
        k_prev_words = new_word_inds[incomplete_inds] # (k)

        step += 1

    # Calculate max score/word
    max_i = complete_seqs_avg_scores.index(max(complete_seqs_avg_scores))
    pred_ind = complete_seqs[max_i]
    alphas = complete_seqs_alpha[max_i]
        
    return pred_ind, alphas


def test(test_loader, encoder, row_encoder, decoder, vocab, beam_size):
    bleu = AverageMeter()
    vocab_size = len(vocab)
    for i, (img, form, form_len) in enumerate(tqdm(test_loader, unit="test_batch")):
        img = img.to(device)
        form = form.to(device)
        form_len = form_len.to(device)

        encoded_img = encoder(img)
        if row_encoder is not None:
            encoded_img = row_encoder(encoded_img)

        pred_ind, alphas = beam_search(encoded_img, encoder, row_encoder, decoder, vocab_size, beam_size)
        predictions = idx2formulas([pred_ind[1:]], vocab) # (1, pred_length)
        references = idx2formulas(to_numpy(form[:, 1:]), vocab) # (1, ref_length)

        assert(len(predictions) == len(references) == 1)

        # Sample prediction in wandb
        pred_str = "".join(predictions[0])
        true_str = "".join(references[0])
        wandb.log({"test_pred_examples": [wandb.Image(img[0], caption="pred: {}\ntrue: {}".format(pred_str, true_str))]})

        # Calculate BLEU score
        for ref, pred in zip(references, predictions):
            bleu_score = sentence_bleu([ref], pred)
        bleu.update(bleu_score)

        # print every 100 batches
        if i % 100 == 0:
            wandb.log({"Test_BLEU":bleu_score})
            print('Test: [{0}/{1}]\t'
                  'BLEU {2} ({bleu.avg:.3f})'.format(i, len(test_loader), bleu_score, bleu=bleu))

    return bleu.avg



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluating...")
    parser.add_argument("--sample", action="store_true", default=False, help="Use sample data or not")
    parser.add_argument("--model_name", type=str, default="BEST.pth.tar", help="Which checkpoint to use")
    parser.add_argument('--beam_size', type=int, default=5, help='Beam size for beam search')
    parser.add_argument("--checkpoint_folder", type=str, default="checkpoint", help="Specify the checkpoint folder path")
    args = parser.parse_args()

    vocab = load_vocab()
    vocab_size = len(vocab)

    print("Loading test data...")
    # Referenced from https://pytorch.org/vision/stable/models.html
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])
    normalize = None
    test_loader = LatexDataloader("test", transform=normalize, batch_size=1, shuffle=True, sample=args.sample)

    print("Loading model...")
    encoder, row_encoder, decoder = load_model(args.checkpoint_folder, vocab_size, args.model_name, args.sample)

    wandb.init(project="im2latex", config=args)
    avg_bleu = test(test_loader=test_loader,
                    encoder=encoder,
                    row_encoder=row_encoder,
                    decoder=decoder, 
                    vocab = vocab,
                    beam_size = args.beam_size)

    print("\nAverage BLEU of the test set is {}. \n".format(avg_bleu))

