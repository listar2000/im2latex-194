from config import PROCESSED_FOLDER_PATH, train_config
from data import Im2LatexDataset
from encoder import Encoder
from decoder import DecoderWithAttention
from build_vocab import load_vocab, Vocab
from utils import *

import time
from nltk.translate.bleu_score import corpus_bleu
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence

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

def load_model(vocab_size):
    encoder = Encoder()
    encoder.fine_tune()
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=train_config["lr"])

    decoder = DecoderWithAttention(vocab_size=vocab_size)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=train_config["lr"])

    encoder = encoder.to(device)
    decoder = decoder.to(device)
    return encoder, decoder, encoder_optimizer, decoder_optimizer

def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch):
    encoder.train()
    decoder.train()

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

        # Forward proprgation
        img = encoder(img)
        scores, form_sorted, decode_lengths, alphas, sort_ind = decoder(img, form, form_len)
        targets = form_sorted[:, 1:]

        print("not padded:", scores.shape, targets.shape, decode_lengths)

        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

        print("packed padded:", scores.shape, targets.shape)

        # Calculate loss
        loss = criterion(scores, targets)
        # Add doubly stochastic attention regularization
        loss += train_config['alpha_c'] * ((1. - alphas.sum(dim=1)) ** 2).mean()

        # Back propagation
        decoder_optimizer.zero_grad()
        encoder_optimizer.zero_grad()
        loss.backward()

        # Update weights
        encoder_optimizer.step()
        decoder_optimizer.step()

        # Keep track of metrics
        top5 = accuracy(scores, targets, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)
        start = time.time()

        # Print every 100 batches
        if i % 100 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses,
                                                                          top5=top5accs))
        

def validate(val_loader, encoder, decoder, criterion, vocab):
    decoder.eval()
    encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()
    start = time.time()

    references = list() # True formulas
    predictions = list() # Predictions
    with torch.no_grad():
        for i, (img, form, formlen) in enumerate(train_loader):
            # Move to GPU, if available
            img = img.to(device)
            form = form.to(device)
            formlen = formlen.to(device)

            # Forward proprgation
            img = encoder(img)
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores_copy = scores.clone()
            scores, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
            targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)

            # Calculate loss
            loss = criterion(scores, targets)
            # Add doubly stochastic attention regularization
            loss += train_config['alpha_c'] * ((1. - alphas.sum(dim=1)) ** 2).mean()

            # Keep track of metrics
            losses.update(loss.item(), sum(decode_lengths))
            top5 = accuracy(scores, targets, 5)
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()

            # print every 100 batches
            if i % 100 == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time,
                                                                                loss=losses, top5=top5accs))


            # References
            true_form = [w for w in form if w not in {vocab["<start>"], vocab["<pad>"]}]
            references.append(true_form)

            # Predictions
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
            preds = temp_preds
            predictions.extend(preds)

            assert len(references) == len(predictions)

        # Calculate BLEU-4 scores
        bleu = corpus_bleu(references, predictions)
        print("BLEU:", bleu)
    return bleu

if __name__ == '__main__':
    vocab = load_vocab()
    vocab_size = len(vocab)
    use_cuda = train_config['use_cuda']
    print("Loading data...")
    train_loader, val_loader = load_data()

    print("Loading model...")
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    encoder, decoder, encoder_optimizer, decoder_optimizer = load_model(vocab_size)
    loss = nn.CrossEntropyLoss().to(device)

    best_bleu = 0
    for epoch in range(0, 1): # train_config["max_epoch"]
        train(train_loader, 
            encoder=encoder, 
            decoder=decoder, 
            criterion=loss,
            encoder_optimizer=encoder_optimizer,
            decoder_optimizer=decoder_optimizer,
            epoch=epoch)
        curr_bleu = validate(val_loader=val_loader,
                            encoder=encoder,
                            decoder=decoder,
                            criterion=criterion, vocab = vocab)
        # Check if there was an improvement
        is_best = curr_bleu > best_bleu
        best_bleu4 = max(curr_bleu, best_bleu)
        if not is_best:
            epochs_since_improvement += 1
            print("\nNo improvement yet. Epochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0




