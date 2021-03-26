from os.path import join
import pickle
import json

from torch.utils.data import Dataset
import torch


class Im2LatexDataset(Dataset):
    def __init__(self, data_dir, split):
        """args:
        data_dir: root dir storing the prepoccessed data
        split: train, validate or test
        """
        assert split in ["train", "validate", "test"]
        self.data_dir = data_dir
        self.split = split
        
        # Load in images
        with open(join(self.data_dir, "{}_IMAGES.pkl".format(self.split)), "rb") as p:
            self.images = pickle.load(p)
        # Load in formulas
        with open(join(self.data_dir, self.split + '_CAPTIONS.json'), 'r') as j:
            self.formulas = json.load(j)

    def __getitem__(self, index):
        img = torch.FloatTensor(self.images[index] / 255.)
        formula, form_len = self.formulas[index]
        formula = torch.LongTensor(formula)
        form_len = torch.LongTensor([form_len])
        return img, formula, form_len

    def __len__(self):
        return len(self.formulas)