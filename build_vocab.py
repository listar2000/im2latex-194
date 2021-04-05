"""
Adapted from:
https://github.com/luopeixiang/im2latex/blob/master/build_vocab.py
"""

import argparse
from config import DATA_FOLDER_PATH, PROCESSED_FOLDER_PATH, preprocess_config
from os.path import join
from collections import Counter
import pickle as pkl

START_TOKEN = 0
PAD_TOKEN = 1
END_TOKEN = 2
UNK_TOKEN = 3

class Vocab(object):
    def __init__(self):
        self.sign2id = {"<start>": START_TOKEN, "<pad>": PAD_TOKEN, 
                        "<end>": END_TOKEN, "<unk>": UNK_TOKEN}
        self.id2sign = dict((idx, token)
                            for token, idx in self.sign2id.items())
        self.length = 4
        self.freqs = [1] * 4

    def add_sign(self, sign, count):
        if sign not in self.sign2id:
            self.sign2id[sign] = self.length
            self.id2sign[self.length] = sign
            self.freqs.append(count)
            self.length += 1

    def __len__(self):
        return self.length


def build_vocab(formulas_path, train_filter_path, min_freq = preprocess_config['min_freq']):
    """
    traverse training formulas to make vocab
    and store the vocab in the file
    """
    vocab = Vocab()
    counter = Counter()

    formulas_file = join(DATA_FOLDER_PATH, formulas_path)
    with open(formulas_file, 'r') as f:
        formulas = [formula.strip('\n') for formula in f.readlines()]

    with open(join(DATA_FOLDER_PATH, train_filter_path), 'r') as f:
        for line in f:
            _, idx = line.strip('\n').split()
            idx = int(idx)
            formula = formulas[idx].split()
            counter.update(formula)

    for word, count in counter.most_common():
        if count >= min_freq:
            vocab.add_sign(word, count)

    vocab_file = join(PROCESSED_FOLDER_PATH, 'vocab.pkl')
    print("Writing Vocab File in ", vocab_file)
    with open(vocab_file, 'wb') as w:
        pkl.dump(vocab, w)


def load_vocab():
    with open(join(PROCESSED_FOLDER_PATH, 'vocab.pkl'), 'rb') as f:
        vocab = pkl.load(f)
    print("=> Load vocab including {} words!".format(len(vocab)))
    return vocab


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Building vocab for Im2Latex...")
    parser.add_argument("--sample", action="store_true", default=False, help="Use sample data or not")
    args = parser.parse_args()
    if args.sample:
        formulas_path = "formulas.norm.lst"
        train_filter_path = "train_filter.lst"
    else:
        formulas_path = "im2latex_formulas.norm.lst"
        train_filter_path = "im2latex_train_filter.lst"
    build_vocab(formulas_path, train_filter_path)