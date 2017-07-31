#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 14:47:42 2016

@author: tomas
"""

import os
import json
import string
from Queue import Queue
from threading import Thread, Lock
import argparse
from math import floor

import h5py
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte
from scipy.misc import imresize

import misc.dataset_loader as dl
from misc.embeddings import dct
import misc.utils as utils
    
"""
{
  "id": [int], Unique identifier for this image,
  "regions": [
    {
      "id": [int] Unique identifier for this region,
      "image": [int] ID of the image to which this region belongs,
      "height": [int] Height of the region in pixels,
      "width": [int] Width of the region in pixels,
      "label": [string] label for this region,
      "x": [int] x-coordinate of the upper-left corner of the region,
      "y": [int] y-coordinate of the upper-left corner of the region,
    },
    ...
  ]
}

The output JSON file is an object with the following elements:
- token_to_idx: Dictionary mapping strings to integers for encoding tokens, 
                in 1-indexed format.
- wtoi: maps words to integer
- itow: Inverse of the above.

The output HDF5 file has the following format to describe N images with
M total regions:

- images: uint8 array of shape (N, 1, image_size, image_size) of pixel data,
  in BDHW format. Images will be resized so their longest edge is image_size
  pixels long, aligned to the upper left corner, and padded with zeros.
  The actual size of each image is stored in the image_heights and image_widths
  fields.
- image_heights: int32 array of shape (N,) giving the height of each image.
- image_widths: int32 array of shape (N,) giving the width of each image.
- original_heights: int32 array of shape (N,) giving the original height of
  each image.
- original_widths: int32 array of shape (N,) giving the original width of
  each image.
- boxes: int32 array of shape (M, 4) giving the coordinates of each bounding box.
  Each row is (xc, yc, w, h) where yc and xc are center coordinates of the box,
  and are one-indexed.
- lengths: int32 array of shape (M,) giving lengths of label sequence for each box
- labels: int32 array of shape (M,) giving the integer label for each region.
- dct_word_embeddings: float32 array of shape (M, 108). DCToW Embedding of the ground truth
  label for a ground truth box
- phoc_word_embeddings: float32 array of shape (M, 540). PHOC Embedding of the ground truth
  label for a ground truth box
- img_to_first_box: int32 array of shape (N,). If img_to_first_box[i] = j then
  labels[j] and boxes[j] give the first annotation for image i
  (using one-indexing).
- img_to_last_box: int32 array of shape (N,). If img_to_last_box[i] = j then
  labels[j] and boxes[j] give the last annotation for image i
  (using one-indexing).
- box_to_img: int32 array of shape (M,). If box_to_img[i] = j then then
  regions[i] and labels[i] refer to images[j] (using one-indexing).
  - region_proposals: int32 array of shape (R, 4) giving the coordinates of each region proposal.
  Each row is (xc, yc, w, h) where yc and xc are center coordinates of the box, and are one-indexed.
  - img_to_first_rp: int32 array of shape (N,). The same as img_to_first_box but for region proposals
  instead of ground truth boxes.
  - img_to_last_rp: int32 array of shape (N,). The same as img_to_last_box but for region proposals
  instead of ground truth boxes.
"""

def build_vocab_dict(vocab):
  token_to_idx, idx_to_token = {}, {}
  next_idx = 1
  for token in vocab:
    token_to_idx[token] = next_idx
    idx_to_token[next_idx] = token
    next_idx = next_idx + 1
    
  return token_to_idx, idx_to_token

def encode_word_embeddings(data, wtoe):
    """
    Encode each label as a word embedding
    """
    we = []
    for datum in data:
        for r in datum['regions']:
            we.append(wtoe[r['label']])
            
    return np.array(we)
    
def encode_labels(data, wtoi):
    """
    Encode each label as an integer
    """
    labels = []
    for datum in data:
        for r in datum['regions']:
            labels.append(wtoi[r['label']])
            
    return np.array(labels)

def encode_boxes(data, original_heights, original_widths, image_size, max_image_size, box_type='gt_boxes'):
    all_boxes = []
    xwasbad = 0
    ywasbad = 0
    wwasbad = 0
    hwasbad = 0
    for i, datum in enumerate(data):
        H, W = original_heights[i], original_widths[i]
        scale = float(image_size) / max(H, W)

        #Needed for not so tightly labeled datasets, like washington
        if box_type == 'region_proposals':
            datum[box_type] = utils.pad_proposals(datum[box_type], (H, W), 10)
        
        for box in datum[box_type]:
            x, y = box[0], box[1]
            w, h = box[2] - box[0], box[3] - box[1]
            x, y = round(scale*(x-1)+1), round(scale*(y-1)+1)
            w, h = round(scale*w), round(scale*h)  
          
            # clamp to image
            if x < 1: x = 1
            if y < 1: y = 1
            if x > max_image_size[1] - 1: 
                x = max_image_size[1] - 1
                xwasbad += 1
            if y > max_image_size[0] - 1: 
                y = max_image_size[0] - 1
                ywasbad += 1
            if x + w > max_image_size[1]: 
                w = max_image_size[1] - x
                wwasbad += 1
            if y + h > max_image_size[0]: 
                h = max_image_size[0] - y
                hwasbad += 1
                break
        
            b = np.asarray([x+floor(w/2), y+floor(h/2), w, h], dtype=np.int32) # also convert to center-coord oriented
            assert b[2] > 0 # width height should be positive numbers
            assert b[3] > 0
            all_boxes.append(b)

    print 'number of bad x,y,w,h: ', xwasbad, ywasbad, wwasbad, hwasbad
    return np.vstack(all_boxes)

def build_img_idx_to_box_idxs(data, boxes='regions'):
    img_idx = 1
    box_idx = 1
    num_images = len(data)
    img_to_first_box = np.zeros(num_images, dtype=np.int32)
    img_to_last_box = np.zeros(num_images, dtype=np.int32)
    for datum in data:
        img_to_first_box[img_idx - 1] = box_idx
        for region in datum[boxes]:
            box_idx += 1
        img_to_last_box[img_idx - 1] = box_idx - 1 # -1 to make these inclusive limits
        img_idx += 1
  
    return img_to_first_box, img_to_last_box

def build_filename_dict(data):
    # First make sure all filenames
  
    next_idx = 1
    filename_to_idx, idx_to_filename = {}, {}
    for img in data:
        filename = img['id']
        filename_to_idx[filename] = next_idx
        idx_to_filename[next_idx] = filename
        next_idx += 1
    return filename_to_idx, idx_to_filename

def encode_filenames(data, filename_to_idx):
    filename_idxs = []
    for img in data:
        filename = img['id']
        idx = filename_to_idx[filename]
    for region in img['regions']:
        filename_idxs.append(idx)
    return np.asarray(filename_idxs, dtype=np.int32)

def add_images(data, h5_file, image_size, max_image_size, num_workers=5):
    num_images = len(data)
    shape = (num_images, 1, max_image_size[0], max_image_size[1])
    image_dset = h5_file.create_dataset('images', shape, dtype=np.uint8)
    original_heights = np.zeros(num_images, dtype=np.int32)
    original_widths = np.zeros(num_images, dtype=np.int32)
    image_heights = np.zeros(num_images, dtype=np.int32)
    image_widths = np.zeros(num_images, dtype=np.int32)
  
    lock = Lock()
    q = Queue()

    for i, img in enumerate(data):
        q.put((i, img['id']))
    
    def worker():
        while True:
            i, filename = q.get()
            img = imread(filename)
            if img.ndim == 3:
                img = img_as_ubyte(rgb2gray(img))
            H0, W0 = img.shape[0], img.shape[1]
            img = imresize(img, float(image_size) / max(H0, W0))

            H, W = img.shape[0], img.shape[1]
            img = np.invert(img)

            lock.acquire()
            if i % 1000 == 0:
                print 'Writing image %d / %d' % (i, len(data))
            original_heights[i] = H0
            original_widths[i] = W0
            image_heights[i] = H
            image_widths[i] = W
            image_dset[i, :, :H, :W] = img
            lock.release()
            q.task_done()
      
    print('adding images to hdf5.... (this might take a while)')
    for i in xrange(num_workers):
        t = Thread(target=worker)
        t.daemon = True
        t.start()
    q.join()

    h5_file.create_dataset('image_heights', data=image_heights)
    h5_file.create_dataset('image_widths', data=image_widths)
    h5_file.create_dataset('original_heights', data=original_heights)
    h5_file.create_dataset('original_widths', data=original_widths)

def encode_splits(data):
  """ Encode splits as intetgers and return the array. """
  lookup = {'train': 0, 'val': 1, 'test': 2}
  return [lookup[datum['split']] for datum in data]

#reset = False
#dataset = 'washington'
#root = 'data/dbs/'
#fold = 1
#augment = False
#suffix = ''

def create_dataset(dataset, root, suffix='', augment=False, fold=1, reset=False):
    num_workers = 5
    image_size = 1720
    alphabet = string.ascii_lowercase + string.digits
    
    if not os.path.exists(root):
        os.makedirs(root)
    
    dataset_full = dataset + '_fold%d' % fold
    outdir = root + dataset_full + '/'
    h5_output = root + dataset_full
    json_output = root + dataset_full
    
    if suffix:
        h5_output += '_' + suffix
        json_output += '_' + suffix

    h5_output += '.h5'
    json_output += '.json'
        
    # read in the data
    data = getattr(dl, 'load_' + dataset)(fold)

    sizes = []
    means = []
    for datum in data:
        img = imread(datum['id'])
        if img.ndim == 3:
            img = img_as_ubyte(rgb2gray(img))
            
        if datum['split'] == 'train':
            means.append(np.invert(img).mean())
        sizes.append(img.shape)
    image_mean = np.mean(means)
    sizes = np.array(sizes)
    max_image_size = sizes.max(axis=0)
    
    if augment:
        num_images = 5000
        num_train = len([datum for datum in data if datum['split'] == 'train'])
        tparams = {}
        # get approximately the same amount of images
        tparams['samples_per_image'] = int(np.round(float(num_images / 2) / num_train)) 
        tparams['shear'] = (-5, 30)
        tparams['order'] = 1            #bilinear
        tparams['selem_size'] = (3, 4)  #max size for square kernel for erosion, dilation
        inplace_data = dl.inplace_augment(data, outdir, tparams=tparams, reset=reset) #original data is kept here
        
        nps = num_images - tparams['samples_per_image'] * num_train
        full_page_data = dl.fullpage_augment(data, outdir, nps, reset=reset) #only augmented data is added here
        data = inplace_data + full_page_data 
    
    # create the output hdf5 file handle
    f = h5py.File(h5_output, 'w')
    
    # add several fields to the file: images, and the original/resized widths/heights
    add_images(data, f, image_size, max_image_size, num_workers)
    f.create_dataset('image_mean', data=np.array([image_mean]))
    
    # add split information
    split = encode_splits(data)
    f.create_dataset('split', data=split)
    
    # build vocabulary
    vocab, _ = utils.build_vocab(data)
    wtoi, itow = build_vocab_dict(vocab) # both mappings are dicts
    
    # encode dct embeddings
    dct_wtoe = {w:dct(w, 3, alphabet) for w in vocab}
    dct_word_embeddings = encode_word_embeddings(data, dct_wtoe)
    f.create_dataset('dct_word_embeddings', data=dct_word_embeddings)
          
    # encode boxes
    original_heights = np.asarray(f['original_heights'])
    original_widths = np.asarray(f['original_widths'])
    gt_boxes = encode_boxes(data, original_heights, original_widths, image_size, max_image_size)
    f.create_dataset('boxes', data=gt_boxes)
    
    # write labels
    labels = encode_labels(data, wtoi)
    f.create_dataset('labels', data=labels)
    
    # integer mapping between image ids and region_proposals ids
    utils.filter_region_proposals(data, original_heights, original_widths, image_size)
    
    region_proposals = encode_boxes(data, original_heights, original_widths, 
                                    image_size, max_image_size, 'region_proposals')
    f.create_dataset('region_proposals', data=region_proposals)
    img_to_first_rp, img_to_last_rp = build_img_idx_to_box_idxs(data, 'region_proposals')
    f.create_dataset('img_to_first_rp', data=img_to_first_rp)
    f.create_dataset('img_to_last_rp', data=img_to_last_rp)
    
    # integer mapping between image ids and box ids
    img_to_first_box, img_to_last_box = build_img_idx_to_box_idxs(data)
    f.create_dataset('img_to_first_box', data=img_to_first_box)
    f.create_dataset('img_to_last_box', data=img_to_last_box)
    
    # filename_to_idx, idx_to_filename = build_filename_dict(data)
    # box_to_img = encode_filenames(data, filename_to_idx)
    # f.create_dataset('box_to_img', data=box_to_img)
    f.close()
    
    # and write the additional json file 
    json_struct = {
        'wtoi': wtoi,
        'itow': itow}
        # 'filename_to_idx': filename_to_idx,
        # 'idx_to_filename': idx_to_filename}
      
    with open(json_output, 'w') as f:
        json.dump(json_struct, f)

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('--root_dir',
    default='data/dbs/',
    help='Path to where your data is located')
    parser.add_argument('--cross_val',
    default=0,
    help='Whether or not to use 4-fold cross validation, default 0')
    args = parser.parse_args()

    folds = 1
    if args.cross_val:
        folds = 4

    for fold in range(1, folds + 1):
        create_dataset('washington', args.root_dir, fold=fold)
        create_dataset('washington', args.root_dir, suffix='augmented', augment=True, fold=fold)
