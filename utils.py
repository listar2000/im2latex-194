import torch
from build_vocab import START_TOKEN, PAD_TOKEN, END_TOKEN
from os.path import join
import os
import math

def collate_fn(batch):
    shape = batch[0][0].shape
    batch = [img_form for img_form in batch if img_form[0].shape == shape]

    # sort by the length of formula
    batch.sort(key=lambda img_form: len(img_form[1]), reverse=True)

    imgs, formulas, formula_lens = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    formulas = formulas2tensor(formulas)
    formula_lens = torch.stack(formula_lens, dim=0)
    return imgs, formulas, formula_lens

def formulas2tensor(formulas):
    """convert formula to tensor"""
    batch_size = len(formulas)
    max_len = len(formulas[0]) # Note: formulas is in desc order of formula length
    tensors = torch.ones(batch_size, max_len, dtype=torch.long) * PAD_TOKEN
    for i, formula in enumerate(formulas):
        for j, form_id in enumerate(formula):
            if j == len(formula)-1:
                assert form_id == END_TOKEN
            tensors[i][j] = form_id
    return tensors

class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.
    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)

def idx2formulas(indices, vocab):
    """
    Convert a list of indices list into latex formulas according to "vocab".
    """
    formulas = []
    for ind_list in indices: # Iterate over each batch
        form = []
        for i in ind_list:
            if i == START_TOKEN:
                continue
            elif i != END_TOKEN:
                form.append(vocab.id2sign[i])
            else:
                break
        formulas.append(form)
    return formulas

def save_checkpoint(folder_name, epoch, epochs_since_improvement, encoder, row_encoder, decoder, encoder_optimizer, row_encoder_optimizer,
                        decoder_optimizer, curr_bleu, is_best, sample=False):
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'bleu': curr_bleu,
             'encoder': encoder.state_dict(),
             'decoder': decoder.state_dict(),
             'encoder_optimizer': encoder_optimizer.state_dict(),
             'decoder_optimizer': decoder_optimizer.state_dict()}
    if row_encoder is not None:
        state["row_encoder"] = row_encoder.state_dict()
        state["row_encoder_optimizer"] = row_encoder_optimizer.state_dict()
    filename = 'epoch_' + str(epoch) + '.pth.tar'
    if sample:
        folder_name = folder_name+"/sample"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    torch.save(state, join(folder_name, filename))
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, join(folder_name, 'BEST.pth.tar'))

def cal_epsilon(k, step, method):
    """
    Reference:
        Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks
        See details in https://arxiv.org/pdf/1506.03099.pdf
    """
    assert method in ['inv_sigmoid', 'exp', 'teacher_forcing']

    if method == 'exp':
        return k**step
    elif method == 'inv_sigmoid':
        return k/(k+math.exp(step/k))
    else:
        return 1.

