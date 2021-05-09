"""
Adapted from 
https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning/blob/master/train.py

"""
import argparse
from config import PROCESSED_FOLDER_PATH, train_config
from dataset import LatexDataloader
from build_vocab import load_vocab, Vocab, START_TOKEN, PAD_TOKEN

from models.encoder import Encoder
from models.row_encoder import RowEncoder
from models.decoder import DecoderWithAttention

import time
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu
import torch
from torch import nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence
from torch_utils import to_numpy, device
import torchvision.transforms as transforms
from utils import *
import wandb

total_step = 0

def load_data(sample=False, normalize=False):
    # Referenced from https://pytorch.org/vision/stable/models.html
    if normalize:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    else:
        normalize = None
    train_loader = LatexDataloader("train", transform=normalize, batch_size=train_config["batch_size"], shuffle=True, sample=sample)
    val_loader = LatexDataloader("validate", transform=normalize, batch_size=train_config["batch_size"], shuffle=True, sample=sample)
    return train_loader, val_loader

def load_model(vocab_size, row, sample, model_folder, model_name):
    if model_name is None:
        encoder = Encoder()
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
        decoder_optimizer = optim.Adam(decoder.parameters(), lr=train_config["lr"])
    
    else:
        # Load checkpoint
        if sample:
            model_folder = model_folder+"/sample"
        model_path = join(model_folder, model_name)
        checkpoint = torch.load(model_path)
        print("Checkpoint ended in epoch {}".format(checkpoint['epoch']))
        
        encoder = Encoder()
        encoder.load_state_dict(checkpoint['encoder'])
        encoder_optimizer = optim.Adam(encoder.parameters())
        encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer'])

        if "row_encoder" in checkpoint:
            print("Using row encoder!")
            row_encoder = RowEncoder()
            row_encoder.load_state_dict(checkpoint['row_encoder'])
            row_encoder.to(device)
            row_encoder_optimizer = optim.Adam(row_encoder.parameters())
            row_encoder_optimizer.load_state_dict(checkpoint['row_encoder_optimizer'])

        else:
            print("Not using row encoder!")
            row_encoder = None
            row_encoder_optimizer = None

        decoder = DecoderWithAttention(vocab_size=vocab_size)
        decoder.load_state_dict(checkpoint['decoder'])
        decoder_optimizer = optim.Adam(decoder.parameters())
        decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer'])

    encoder = encoder.to(device)
    decoder = decoder.to(device)

    return encoder, row_encoder, decoder, encoder_optimizer, row_encoder_optimizer, decoder_optimizer

def train(epoch, train_loader, criterion, 
          encoder, encoder_optimizer,
          decoder, decoder_optimizer, 
          row_encoder=None, row_encoder_optimizer=None):
    
    global total_step
    # Set models to training mode.
    encoder.train()
    if row_encoder is not None:
        row_encoder.train()
    decoder.train()

    decay_k = train_config['decay_k']
    decay_method = train_config['decay_method']

    # Initialize performance metrics.
    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy

    start = time.time()
    with tqdm(train_loader, unit="batch") as tepoch:
        for i, (img, form, form_len) in enumerate(tepoch):
            total_step += 1

            tepoch.set_description(f"Epoch {epoch}")
            data_time.update(time.time() - start)

            # Move to GPU, if available
            img = img.to(device)
            form = form.to(device)
            form_len = form_len.to(device)

            # Forward proprogation
            encoded_img = encoder(img)
            if row_encoder is not None:
                encoded_img = row_encoder(encoded_img)

            curr_eps = cal_epsilon(decay_k, total_step, decay_method)  
            scores, alphas = decoder(encoded_img, form, form_len, curr_eps) 

            # Remove <start> token   
            targets = form[:, 1:]

            # A convenient way of removing <pad> tokens
            decode_lengths = form_len+1 # Length of the original sequence, not including start or padding
            scores = pack_padded_sequence(scores, decode_lengths.cpu(), batch_first=True).data.to(device)
            targets = pack_padded_sequence(targets, decode_lengths.cpu(), batch_first=True).data.to(device)

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

            # Print every 100 batches
            if i % 100 == 0:
                wandb.log({"Epoch": epoch, 
                           "Memory allocation(GB)": torch.cuda.memory_allocated(device)/(1024**3),
                           "Train_loss":losses.val,
                           "Train_avg_loss:":losses.avg,
                           "Train_top5_acc": top5accs.val,
                           "Decayed_eps": curr_eps})
                print('Memory allocation: {:.3f}GB\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(torch.cuda.memory_allocated(device)/(1024**3),
                                                                              batch_time=batch_time,
                                                                              data_time=data_time, loss=losses,
                                                                              top5=top5accs))


def validate(epoch, val_loader, encoder, row_encoder, decoder, criterion, vocab):
    decoder.eval()
    encoder.eval()
    if row_encoder is not None:
        row_encoder.eval()

    # Initialize performance metrics.
    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()
    bleu = AverageMeter()
    start = time.time()

    references = list() # True formulas
    predictions = list() # Predictions
    with torch.no_grad():
        for i, (img, form, form_len) in enumerate(tqdm(val_loader, unit="val_batch")):
            # Move to GPU, if available
            img = img.to(device)
            form = form.to(device)
            form_len = form_len.to(device)

            # print("Original img shape:", img.shape)

            # Forward proprgation
            encoded_img = encoder(img)
            if row_encoder is not None:
                encoded_img = row_encoder(encoded_img)
            # Set epsilon=0 (Not using teacher forcing for validation)
            scores, alphas = decoder(encoded_img, form, form_len, epsilon=0)
            scores_copy = scores.clone()

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = form[:, 1:]
            targets_copy = targets.clone()

            # A convenient way of removing <pad> tokens
            decode_lengths = form_len+1 # Length of the original sequence, not including start or padding
            scores = pack_padded_sequence(scores, decode_lengths.cpu(), batch_first=True).data.to(device)
            targets = pack_padded_sequence(targets, decode_lengths.cpu(), batch_first=True).data.to(device)

            # Calculate loss
            loss = criterion(scores, targets)
            # Add doubly stochastic attention regularization
            loss += train_config['alpha_c'] * ((1. - alphas.sum(dim=1)) ** 2).mean()

            # Update performance metrics
            losses.update(loss.item(), sum(decode_lengths))
            top5 = accuracy(scores, targets, 5)
            top5accs.update(top5, sum(decode_lengths))
            
            _, preds = torch.max(scores_copy, dim=2) # return indices of max scores 
            predictions = idx2formulas(to_numpy(preds), vocab)
            references = idx2formulas(to_numpy(targets_copy), vocab)

            # Sample prediction in wandb
            pred_str = "".join(predictions[0])
            true_str = "".join(references[0])
            wandb.log({"val_pred_examples": [wandb.Image(img[0], caption="pred: {}\ntrue: {}".format(pred_str, true_str))]})

            # Calculate BLEU score
            total_score = 0.0
            for ref, pred in zip(references, predictions):
                bleu_score = sentence_bleu([ref], pred)
                total_score += bleu_score
                bleu.update(bleu_score)
            

            batch_time.update(time.time() - start)
            # print every 100 batches
            if i % 100 == 0:
                wandb.log({"Epoch":epoch,
                           "Val_loss":losses.val,
                           "Val_avg_loss:":losses.avg,
                           "Val_top5_acc": top5accs.val,
                           "Val_avg_BLEU": bleu.avg})
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'
                      'Average BLEU {2} ({bleu.avg:.3f})'.format(i, len(val_loader), total_score/len(predictions), 
                                                                 batch_time=batch_time,
                                                                 loss=losses, 
                                                                 top5=top5accs,
                                                                 bleu=bleu))
            
            start = time.time()                                                               

            # # References
            # targets_copy = targets_copy.tolist()
            # refs = idx2formulas(targets_copy, vocab)
            # refs = [[x] for x in refs]
            # references.append(refs)

            # # Predictions
            # _, preds = torch.max(scores_copy, dim=2) # return indices of max scores
            # preds = preds.tolist()
            # preds = idx2formulas(preds, vocab)
            # predictions.append(preds)

            # assert len(refs) == len(preds)

        # Calculate BLEU-4 scores
        # print("Ref:")
        # print(references)
        # print("\n")
        # print("Pred:")
        # print(predictions)

        # bleu = corpus_bleu(references, predictions)
        # print("BLEU-4:", bleu)
    return bleu.avg

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training...")
    parser.add_argument("--sample", action="store_true", default=False, help="Use sample data or not")
    parser.add_argument("--row", action="store_true", default=False, help="Use row_encoder or not")
    parser.add_argument('--max_epoch', type=int, default=1, help='max epoch for training')
    parser.add_argument("--checkpoint_folder", type=str, default="checkpoint", help="Specify the checkpoint folder path")
    parser.add_argument("--model_name", type=str, default=None, help="Which checkpoint to use")
    args = parser.parse_args()

    wandb.init(project="im2latex", config=train_config)
    wandb.config.update(args)

    vocab = load_vocab()
    vocab_size = len(vocab)

    print("Loading data...")
    train_loader, val_loader = load_data(args.sample)

    print("Loading model...")
    encoder, row_encoder, decoder, encoder_optimizer, row_encoder_optimizer, decoder_optimizer = load_model(vocab_size, 
                                                                                                            row=args.row,
                                                                                                            sample=args.sample, 
                                                                                                            model_folder=args.checkpoint_folder, 
                                                                                                            model_name=args.model_name)
    criterion = nn.CrossEntropyLoss().to(device)

    wandb.watch(encoder)
    if row_encoder is not None:
        wandb.watch(row_encoder)
    wandb.watch(decoder)

    best_bleu = 0
    start_epoch = 0
    epochs_since_improvement = 0
    if args.model_name:
        # Load checkpoint info
        model_folder = args.checkpoint_folder
        if args.sample:
            model_folder = model_folder+"/sample"
        model_path = join(model_folder, args.model_name)
        checkpoint = torch.load(model_path)
        start_epoch = checkpoint['epoch']+1
        best_bleu = checkpoint['bleu']

    for epoch in range(start_epoch, args.max_epoch):
        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == 20:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(encoder_optimizer, 0.8)
            if row_encoder is not None:
                adjust_learning_rate(row_encoder_optimizer, 0.8)
            adjust_learning_rate(decoder_optimizer, 0.8)

        train(train_loader=train_loader, 
              encoder=encoder, 
              row_encoder=row_encoder,
              decoder=decoder, 
              criterion=criterion,
              encoder_optimizer=encoder_optimizer,
              row_encoder_optimizer=row_encoder_optimizer,
              decoder_optimizer=decoder_optimizer,
              epoch=epoch)

        print("\n========== Validating ==========")
        curr_bleu = validate(val_loader=val_loader,
                            encoder=encoder,
                            row_encoder=row_encoder,
                            decoder=decoder,
                            criterion=criterion, 
                            vocab = vocab,
                            epoch=epoch)

        # Check if there was an improvement
        is_best = curr_bleu > best_bleu
        best_bleu = max(curr_bleu, best_bleu)
        if not is_best:
            epochs_since_improvement += 1
            print("\nNo improvement yet. Epochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint every epoch
        save_checkpoint(args.checkpoint_folder, epoch, epochs_since_improvement, encoder, row_encoder, decoder, encoder_optimizer, 
                            row_encoder_optimizer, decoder_optimizer, curr_bleu, is_best, sample=args.sample)





