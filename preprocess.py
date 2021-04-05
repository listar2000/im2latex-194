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
import argparse
from config import DATA_FOLDER_PATH, PROCESSED_FOLDER_PATH, preprocess_config
import h5py
from tqdm import tqdm
from skimage.io import imread
from skimage.transform import resize as imresize
from build_vocab import load_vocab, Vocab
import numpy as np
import json

IMG_SIZES = np.array([(64, 320), (64, 384), (64, 480), \
    (64, 224), (64, 256), (32, 320), (64, 192), \
    (64, 160), (32, 192), (32, 224), (32, 160), \
    (32, 128), (32, 256), (64, 128), (32, 384), \
    (32, 480), (96, 384), (128, 480)])

def preprocess(store_pkl, train=True, val=True, test=True, max_len=preprocess_config['max_len'], sample=False):
    formulas_file_name = "formulas.norm.lst" if sample else "im2latex_formulas.norm.lst"
    formulas_file = join(DATA_FOLDER_PATH, formulas_file_name)
    with open(formulas_file, 'r') as f:
        formulas = [formula.strip('\n') for formula in f.readlines()]
    
    vocab = load_vocab()
    word_map = vocab.sign2id

    prep_fn = preprocess_pkl if store_pkl else preprocess_hdf

    if train:
        prep_fn('train', formulas, word_map, max_len, sample)
    if val:
        prep_fn('validate', formulas, word_map, max_len, sample)
    if test:
        prep_fn('test', formulas, word_map, max_len, sample)

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

def adaptive_transform(img, old_size):
    # using numpy to accelerates
    old_size = np.array(old_size)
    # find the accepted size that is closest to current size in terms of euclidean distance
    best_i = np.argmin(np.sum((IMG_SIZES - old_size) ** 2, axis=1))
    new_size = tuple(IMG_SIZES[best_i])
    img = imresize(img, new_size)
    return img, new_size

def preprocess_hdf(split, formulas, word_map, max_len, sample=False):
    assert split in ["train", "validate", "test"]

    print("*** start preprocessing into the .hdf5 format")
    print("Process {} dataset...".format(split))

    if sample:
        split_fn = "{}_filter.lst".format(split)
    else:
        split_fn = "im2latex_{}_filter.lst".format(split)
    split_fp = join(DATA_FOLDER_PATH, split_fn)
    img_folder_name = "images_processed" if sample else "formula_images_processed"

    img_by_size, formula_by_size = {}, {}
    print("loading images & formula id from {}...".format(split_fn))
    with open(split_fp, 'r') as f:
        for line in tqdm(f):
            img_name, formula_id = line.strip('\n').split()
            cap = formulas[int(formula_id)]
            cap_len = len(cap.split())
            if cap_len <= max_len:
                img = imread(join(join(DATA_FOLDER_PATH, img_folder_name), img_name))
                img_size = tuple(img.shape[:2])
                # update when the size is not in the list
                if img_size not in IMG_SIZES:
                    img, img_size = adaptive_transform(img, img_size)
                img_lst = img_by_size.setdefault(img_size, [])
                fml_lst = formula_by_size.setdefault(img_size, [])

                img = img.transpose(2, 0, 1)
                assert img.shape[0] == 3 and np.max(img) <= 255
                img_lst.append(img)
                fml_lst.append(formula_id)

    # storing images to different bins
    print("writing images to " + str.upper(split) + '_IMAGES' + '.hdf5')
    with h5py.File(join(PROCESSED_FOLDER_PATH, str.upper(split) + '_IMAGES' + '.hdf5'), 'a') as h:
        for i, size in tqdm(enumerate(img_by_size)):
            size_str = "{}x{}".format(size[0], size[1])
            img_lst = img_by_size[size]
            h.create_dataset(size_str, data=img_lst)
            print("==> successfully writing {} images of size {}".format(len(img_lst), size_str))
    
    formula_by_size = {"{}x{}".format(k[0], k[1]):v for (k, v) in formula_by_size.items()}
    print("saving caption indices to {}_CAPTIONS.json".format(str.upper(split)))
    # Save encoded captions and their lengths to JSON files
    with open(join(PROCESSED_FOLDER_PATH, str.upper(split) + '_CAPTIONS' + '.json'), 'w') as j:
        json.dump(formula_by_size, j)

def preprocess_pkl(split, formulas, word_map, max_len, sample=False):
    assert split in ["train", "validate", "test"]

    print("*** start preprocessing into the .pkl format")
    print("Process {} dataset...".format(split))

    if sample:
        split_fn = "{}_filter.lst".format(split)
    else:
        split_fn = "im2latex_{}_filter.lst".format(split)
    split_fp = join(DATA_FOLDER_PATH, split_fn)
    img_folder_name = "images_processed" if sample else "formula_images_processed"

    images, caption_info = [], []

    tot, val = 0, 0
    print("loading images & formula id from {}...".format(split_fn))
    with open(split_fp, 'r') as f:
        for line in tqdm(f):
            tot += 1
            img_name, formula_id = line.strip('\n').split()
            formula = formulas[int(formula_id) - 1]
            form_len = len(formula.split())
            if form_len <= max_len:
                val += 1
                img = imread(join(join(DATA_FOLDER_PATH, img_folder_name), img_name))
                img = img.transpose(2, 0, 1)
                assert img.shape[0] == 3
                assert np.max(img) <= 255
                images.append(img)
                caption_info.append(caption_embed(word_map, formula, max_len))
    
    print("loading completed, scanned a total of {} files, saved {} files with token length < {}"\
        .format(tot, val, max_len))

    out_file = join(PROCESSED_FOLDER_PATH, "{}_IMAGES.pkl".format(str.upper(split)))
    with open(out_file, 'wb') as w:
        pkl.dump(images, w)

    print("saved {} images to {}".format(split, out_file))
    # Save encoded captions and their lengths to JSON files
    with open(join(PROCESSED_FOLDER_PATH, str.upper(split) + '_CAPTIONS' + '.json'), 'w') as j:
        json.dump(caption_info, j)

def img_size(pair):
    img, formula = pair
    return tuple(img.size())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocessing...")
    parser.add_argument("--sample", action="store_true", default=False, help="Use sample data or not")
    args = parser.parse_args()
    preprocess(store_pkl=False, sample=args.sample)
