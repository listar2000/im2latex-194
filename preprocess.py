"""
@author Star Li
@time 3/23/2021

This script is used to preprocess the raw data in DATA_FOLDER_PATH into formats that can be 
directly fed into dataset.py (train/val/test). The two formats supported are:

1) the pkl format, which serializes a python object/dictionary into the hard drive and retrieves
when needed. When loading & training, the around 1GB data must be loaded into the memory first.

the implementation for the pkl format directly borrows from 
=> https://github.com/luopeixiang/im2latex/blob/master/build_vocab.py

2) the HDF5 file, which does the pretty much same thing except that now, during training, the 
dataset loads images directly from the hard drive (SSD). This would save memory use at the 
cost of higher loading time (this won't be the performance bottleneck generally).

the implementation is inspired by a project for the image captioning task at
=> https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning
"""

from os.path import join
import pickle as pkl

from torch.serialization import validate_cuda_device
from config import DATA_FOLDER_PATH, IMG_FOLDER_PATH, PROCESSED_FOLDER_PATH, preprocess_config
import h5py
from tqdm import tqdm
from skimage.io import imread
from skimage.transform import resize as imresize
from build_vocab import load_vocab, Vocab
import numpy as np
import json

def preprocess(store_pkl, train=True, val=True, test=True, max_len=preprocess_config['max_len']):
    formulas_file = join(DATA_FOLDER_PATH, "im2latex_formulas.norm.lst")
    with open(formulas_file, 'r') as f:
        formulas = [formula.strip('\n') for formula in f.readlines()]
    
    vocab = load_vocab()
    word_map = vocab.sign2id

    prep_fn = preprocess_pkl if store_pkl else preprocess_hdf

    if train:
        prep_fn('train', formulas, word_map, max_len)
    if val:
        prep_fn('validate', formulas, word_map, max_len)
    if test:
        prep_fn('test', formulas, word_map, max_len)

def caption_embed(word_map, caption, max_len, padding=preprocess_config['padding']):

    caption = caption.split()
    encoded_cap = [word_map['<start>']]

    for word in caption:
        encoded_cap.append(word_map.get(word, word_map['<unk>']))
    
    encoded_cap.append(word_map['<end>'])

    if padding:
        encoded_cap.extend([word_map['<pad>']] * (max_len - len(caption)))

    # return both the encoded caption (maybe padded) and the original length
    return encoded_cap, len(caption)

def preprocess_hdf(split, formulas, word_map, max_len):
    assert split in ["train", "validate", "test"]

    print("*** start preprocessing into the .hdf5 format")
    print("Process {} dataset...".format(split))

    split_fn = "im2latex_{}_filter.lst".format(split)
    split_fp = join(DATA_FOLDER_PATH, split_fn)

    img_names, formula_ids = [], []
    print("loading file names & formula id from {}...".format(split_fn))
    with open(split_fp, 'r') as f:
        for line in tqdm(f):
            img_name, formula_id = line.strip('\n').split()
            img_names.append(img_name)
            formula_ids.append(int(formula_id))

    caption_info = []
    print("writing images to " + split + '_IMAGES' + '.hdf5')
    with h5py.File(join(PROCESSED_FOLDER_PATH, split + '_IMAGES' + '.hdf5'), 'a') as h:
        images = h.create_dataset('images', (len(img_names), 3, 256, 256), dtype='uint8')
        # this is a list of tuples with (encoded caption, caption length) stored as json file   
        for i, img_name in enumerate(tqdm(img_names)):
            img = imread(join(IMG_FOLDER_PATH, img_name))
            img = imresize(img, (256, 256))
            img = img.transpose(2, 0, 1)
            assert img.shape[0] == 3
            assert np.max(img) <= 255
            # Save image to HDF5 file
            images[i] = img

            caption_info.append(caption_embed(word_map, formulas[formula_ids[i]], max_len))
    
    # Save encoded captions and their lengths to JSON files
    with open(join(PROCESSED_FOLDER_PATH, split + '_CAPTIONS' + '.json'), 'w') as j:
        json.dump(caption_info, j)

def preprocess_pkl(split, formulas, word_map, max_len):
    assert split in ["train", "validate", "test"]

    print("*** start preprocessing into the .pkl format")
    print("Process {} dataset...".format(split))

    split_fn = "im2latex_{}_filter.lst".format(split)
    split_fp = join(DATA_FOLDER_PATH, split_fn)

    images, caption_info = [], []

    tot, val = 0, 0
    print("loading images & formula id from {}...".format(split_fn))
    with open(split_fp, 'r') as f:
        for line in tqdm(f):
            tot += 1
            img_name, formula_id = line.strip('\n').split()
            formula = formulas[int(formula_id)]
            form_len = len(formula.split())
            if form_len <= max_len:
                val += 1
                img = imread(join(IMG_FOLDER_PATH, img_name))
                img = img.transpose(2, 0, 1)
                assert img.shape[0] == 3
                assert np.max(img) <= 255
                images.append(img)
                caption_info.append(caption_embed(word_map, formula, max_len))
    
    print("loading completed, scanned a total of {} files, saved {} files with token length < {}"\
        .format(tot, val, max_len))

    out_file = join(PROCESSED_FOLDER_PATH, "{}_IMAGES.pkl".format(split))
    with open(out_file, 'wb') as w:
        pkl.dump(images, w)

    print("saved {} images to {}".format(split, out_file))
    # Save encoded captions and their lengths to JSON files
    with open(join(PROCESSED_FOLDER_PATH, split + '_CAPTIONS' + '.json'), 'w') as j:
        json.dump(caption_info, j)

def img_size(pair):
    img, formula = pair
    return tuple(img.size())

if __name__ == '__main__':
    preprocess(True)

