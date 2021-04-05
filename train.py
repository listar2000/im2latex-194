"""
Adapted from 
https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning/blob/master/train.py

"""

from config import PROCESSED_FOLDER_PATH, train_config
from data import Im2LatexDataset
from build_vocab import load_vocab, Vocab, START_TOKEN, PAD_TOKEN

from encoder import Encoder
from row_encoder import RowEncoder
from decoder import DecoderWithAttention

import time
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence
from utils import *

use_cuda = train_config['use_cuda']
device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")


def load_data():
    train_loader = DataLoader(Im2LatexDataset(PROCESSED_FOLDER_PATH, 'train'),
        batch_size=train_config['batch_size'],
        shuffle=True,
        collate_fn = collate_fn,
        pin_memory=True if train_config['use_cuda'] else False,
        num_workers=train_config['num_workers'])
    val_loader = DataLoader(Im2LatexDataset(PROCESSED_FOLDER_PATH, 'validate'),
        batch_size=train_config['batch_size'],
        shuffle=True,
        collate_fn = collate_fn,
        pin_memory=True if train_config['use_cuda'] else False,
        num_workers=train_config['num_workers'])
    return train_loader, val_loader

def load_model(vocab_size, row):
    encoder = Encoder()
    encoder = encoder.to(device)
    encoder.fine_tune()
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=train_config["lr"])

    if row:
        row_encoder = RowEncoder()
        row_encoder.to(device)
        row_encoder_optimizer = optim.Adam(row_encoder.parameters(), lr=train_config["lr"])
    else:
        row_encoder = None
        row_encoder_optimizer = None

    decoder = DecoderWithAttention(vocab_size=vocab_size)
    decoder = decoder.to(device)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=train_config["lr"])

    return encoder, row_encoder, decoder, encoder_optimizer, row_encoder_optimizer, decoder_optimizer

def train(epoch, train_loader, criterion, 
          encoder, encoder_optimizer,
          decoder, decoder_optimizer, 
          row_encoder=None, row_encoder_optimizer=None):
    # Set models to training mode.
    encoder.train()
    if row_encoder is not None:
        row_encoder.train()
    decoder.train()

    # Initialize performance metrics.
    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy

    start = time.time()
    for i, (img, form, form_len) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to GPU, if available
        img = img.to(device)
        form = form.to(device)
        form_len = form_len.to(device)

        # Forward proprogation
        img = encoder(img)
        if row_encoder is not None:
            img = row_encoder(img)
        scores, alphas = decoder(img, form, form_len) 

        # Remove <start> token   
        targets = form[:, 1:]

        # A convenient way of removing <end> and <pad> tokens
        decode_lengths = form_len.squeeze(1) # Length of the original sequence, not including start, end, or padding
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

        # Calculate loss
        loss = criterion(scores, targets)
        # Add doubly stochastic attention regularization
        loss += train_config['alpha_c'] * ((1. - alphas.sum(dim=1)) ** 2).mean()

        # Back propagation
        decoder_optimizer.zero_grad()
        if row_encoder_optimizer is not None:
            row_encoder_optimizer.zero_grad()
        encoder_optimizer.zero_grad()
        loss.backward()

        # Update weights
        decoder_optimizer.step()
        if row_encoder_optimizer is not None:
            row_encoder_optimizer.step()
        encoder_optimizer.step()

        # Update performance metrics
        top5 = accuracy(scores, targets, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)
        start = time.time()

        # Print every 5 batches
        if i % 5 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses,
                                                                          top5=top5accs))
        

def validate(val_loader, encoder, row_encoder, decoder, criterion, vocab):
    decoder.eval()
    encoder.eval()
    if row_encoder is not None:
        row_encoder.eval()

    # Initialize performance metrics.
    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()
    start = time.time()

    references = list() # True formulas
    predictions = list() # Predictions
    with torch.no_grad():
        for i, (img, form, form_len) in enumerate(val_loader):
            # Move to GPU, if available
            img = img.to(device)
            form = form.to(device)
            form_len = form_len.to(device)

            # print("Original img shape:", img.shape)

            # Forward proprgation
            img = encoder(img)
            if row_encoder is not None:
                img = row_encoder(img)
            scores, alphas = decoder(img, form, form_len)
            scores_copy = scores.clone()

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = form[:, 1:]
            targets_copy = targets.clone()

            # A convenient way of removing <end> and <pad> tokens
            decode_lengths = form_len.squeeze(1) # Length of the original sequence, not including start, end, or padding
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data
            
            # Calculate loss
            loss = criterion(scores, targets)
            # Add doubly stochastic attention regularization
            loss += train_config['alpha_c'] * ((1. - alphas.sum(dim=1)) ** 2).mean()

            # Update performance metrics
            losses.update(loss.item(), sum(decode_lengths))
            top5 = accuracy(scores, targets, 5)
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()

            # print every 5 batches
            if i % 5 == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time,
                                                                                loss=losses, top5=top5accs))


            # References
            targets_copy = targets_copy.tolist()
            refs = idx2formulas(targets_copy, vocab)
            refs = [[x] for x in refs]
            references.append(refs)

            # Predictions
            _, preds = torch.max(scores_copy, dim=2) # return indices of max scores
            preds = preds.tolist()
            preds = idx2formulas(preds, vocab)
            predictions.append(preds)

            assert len(refs) == len(preds)

        # Calculate BLEU-4 scores
        # print("Ref:")
        # print(references)
        # print("\n")
        # print("Pred:")
        # print(predictions)

        # bleu = corpus_bleu(references, predictions)
        # print("BLEU-4:", bleu)
    return bleu

if __name__ == '__main__':
    vocab = load_vocab()
    vocab_size = len(vocab)
    use_row = train_config['use_row']
    print("Loading data...")
    train_loader, val_loader = load_data()

    print("Loading model...")
    encoder, row_encoder, decoder, encoder_optimizer, row_encoder_optimizer, decoder_optimizer = load_model(vocab_size, row=use_row)
    criterion = nn.CrossEntropyLoss().to(device)

    best_bleu = 0
    for epoch in range(0, train_config["max_epoch"]):
        train(train_loader=train_loader, 
              encoder=encoder, 
              row_encoder=row_encoder,
              decoder=decoder, 
              criterion=criterion,
              encoder_optimizer=encoder_optimizer,
              row_encoder_optimizer=row_encoder_optimizer,
              decoder_optimizer=decoder_optimizer,
              epoch=epoch)
        curr_bleu = validate(val_loader=val_loader,
                            encoder=encoder,
                            row_encoder=row_encoder,
                            decoder=decoder,
                            criterion=criterion, 
                            vocab = vocab)

        # Check if there was an improvement
        is_best = curr_bleu > best_bleu
        best_bleu = max(curr_bleu, best_bleu)
        if not is_best:
            epochs_since_improvement += 1
            print("\nNo improvement yet. Epochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0




