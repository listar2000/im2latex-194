import torch
from build_vocab import PAD_TOKEN, END_TOKEN

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
            if i != END_TOKEN:
                form.append(vocab.id2sign[i])
            else:
                break
        formulas.append(form)
    return formulas

def save_checkpoint(epoch, epochs_since_improvement, encoder, row_encoder, decoder, encoder_optimizer, row_encoder_optimizer,
                        decoder_optimizer, curr_loss, is_best, sample=False):
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'loss': curr_loss,
             'encoder': encoder,
             'row_encoder': row_encoder,
             'decoder': decoder,
             'encoder_optimizer': encoder_optimizer,
             'row_encoder_optimizer': row_encoder_optimizer,
             'decoder_optimizer': decoder_optimizer}
    filename = 'epoch_' + epoch + '.pth.tar'
    folder_name = "checkpoint"
    if sample:
        folder_name = "checkpoint/sample"
    torch.save(state, join(folder_name, filename))
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, join(folder_name, 'BEST_' + filename))



