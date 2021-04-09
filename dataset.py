"""
@author Star Li
@time 4/2/2021
"""

from typing import Iterable, Iterator, Optional
import torch
from torch._C import dtype
from config import DATA_FOLDER_PATH, PROCESSED_FOLDER_PATH
from os.path import join
import h5py, json
import numpy as np
from build_vocab import load_vocab, Vocab

class LatexDataloader(Iterable):
    """
    creating a LatexDataLoader that is iterable (see below for sample usage)
    split: one of the `train`, `validate`, `test`
    transform: optional, recommended to be from torchvision.transforms
    shuffle: whether to shuffle the minibatches (both image bins and images within a bin), default true
    batch_size: size of a minibatch, recommended to be 2**i where i is a natural number

    !! RETURN VALUES: a tuple of size three (images_data, latex_data, latex_lens)
    1. images_data
    a torch float tensor of dimensions N x H x W where N is the batch_size (default 16)
    
    2. latex_data
    a torch int16 tensor of dimensions N x L where L is the longest formula in that batch 
    of size N

    3. latex_lens
    a torch int16 tensor of dimensions N x 1 where each element represents the original 
    length of the text formula (note that the <start> and <end> tokens are not counted) 
    """
    def __init__(self, split: str, transform=None, shuffle: bool=True, batch_size: int=16):
        self.split = str.upper(split)
        assert self.split in {'TRAIN', 'VALIDATE', 'TEST'}

        self.shuffle = shuffle
        self.batch_size = batch_size

        json_path = join(PROCESSED_FOLDER_PATH, "{}_CAPTIONS.json".format(self.split))
        formula_path = join(DATA_FOLDER_PATH, "im2latex_formulas.norm.lst")

        with open(formula_path, 'r') as f:
            self.formulas = [formula.strip('\n') for formula in f.readlines()]

        # loading json file
        with open(json_path, 'r') as j:
            self.formula_dict = json.load(j) 

        # Total number of datapoints
        self.bin_sizes = [len(formulas) for formulas in self.formula_dict.values()]
        self.bin_names = list(self.formula_dict.keys())
        self.dataset_size = sum(self.bin_sizes)
        self.num_bins = len(self.formula_dict)

        hdf_path = join(PROCESSED_FOLDER_PATH, "{}_IMAGES.hdf5".format(self.split))

        self.db = h5py.File(hdf_path, 'r')

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

    def __len__(self):
        return self.dataset_size

    def __iter__(self):
        return LatexDataIterator(self)

class LatexDataIterator(Iterator):

    def __init__(self, loader: LatexDataloader):
        self.loader = loader
        self.bs = self.loader.batch_size
        self._bin_yielding = 0 # the index of the bin currently yielding
        self._num_yielded = 0 # how many items inside a bin has been yielded

        self.bin_order = np.arange(self.loader.num_bins)

        if self.loader.shuffle:
            np.random.shuffle(self.bin_order)
        
        self._switch_bin(self.bin_order[0])
        # loading the vocabulary from preprocessed training formulas
        self.vocab: Vocab = load_vocab()

    def _switch_bin(self, bin_num):
        self.curr_bin_size = self.loader.bin_sizes[bin_num]
        curr_db_name = self.loader.bin_names[bin_num]
        self.curr_db = self.loader.db[curr_db_name]
        self.curr_latex = np.array(self.loader.formula_dict[curr_db_name])
        self._num_yielded = 0

        self.item_order = np.arange(self.curr_bin_size)
        if self.loader.shuffle:
            np.random.shuffle(self.item_order)

    """
    input: a formula 
    """
    def _word_embed(self, formula):
        embedded = [0] # a start
        for word in formula:
            if word in self.vocab.sign2id:
                embedded.append(self.vocab.sign2id[word])
            else:
                embedded.append(3) # unknown
        embedded.append(2)

        return embedded

    def _pad_formulas(self, formulas, latex_lens, latex_order):
        # remember the start & the end added
        max_len = latex_lens[latex_order[0]] + 2
        padded = np.ones((len(latex_lens), max_len), dtype=np.int16)

        for i, o in enumerate(latex_order):
            formula_len = latex_lens[o] + 2 
            padded[i, :formula_len] = formulas[o]  

        return padded

    def __iter__(self):
        return self

    def __next__(self):
        curr_bin_left = self.curr_bin_size - self._num_yielded
        if curr_bin_left == 0:
            self._bin_yielding += 1

            if self._bin_yielding >= len(self.bin_order):
                raise StopIteration("all data used up")

            self._switch_bin(self.bin_order[self._bin_yielding])
            curr_bin_left = self.curr_bin_size - self._num_yielded

        if curr_bin_left < self.bs:
            idx = self.item_order[self._num_yielded:]
            self._num_yielded = self.curr_bin_size 
        else:
            # print(self._num_yielded, self.curr_bin_size)
            idx = self.item_order[self._num_yielded : self._num_yielded + self.bs]
            self._num_yielded += self.bs
        
        if self.loader.shuffle:
            idx = sorted(idx)

        # turn into torch float tensor with value between 0.0 and 1.0
        images_data = torch.FloatTensor(self.curr_db[idx] / 255.)

        if self.loader.transform is not None:
            images_data = self.loader.transform(images_data)

        latex_line_indices = self.curr_latex[idx]
        latex_data, latex_lens = [], []
        for index in latex_line_indices:
            formula = self.loader.formulas[int(index)]
            latex_data.append(self._word_embed(formula))
            latex_lens.append(len(formula))

        latex_lens = np.array(latex_lens, dtype=np.int32)
        latex_order = np.argsort(-latex_lens) # trick to sort descending in numpy

        latex_data = self._pad_formulas(latex_data, latex_lens, latex_order)

        latex_lens = latex_lens[latex_order]
        latex_data = torch.from_numpy(latex_data)
        latex_lens = torch.from_numpy(latex_lens)

        return (images_data, latex_data, latex_lens)


if __name__ == '__main__':
    # import time
    # dataloader = LatexDataloader("validate", batch_size=16, shuffle=True)
    # start = time.time()
    # j = 0
    # for a, b, c in dataloader:
    #     j += len(c)

    #     if j > 30:
    #         break

    # print(time.time() - start)
    # print(j)
    # print(dataloader.dataset_size)
    dataloader = LatexDataloader("validate", batch_size=16, shuffle=True)

    dataIter = iter(dataloader)
    for i in range(5):
        images, latex_data, index_lens = next(dataIter)
        print(index_lens)