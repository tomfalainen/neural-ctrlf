#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 12:10:04 2016

@author: tomas
"""

import os
import glob
import string
import json
import copy
import os.path as osp

import numpy as np
from skimage.io import imread, imsave
import skimage.filters as fi
import skimage.transform as tf
from skimage.util import img_as_ubyte
from skimage.color import rgb2gray
import skimage.morphology as mor
from skimage.filters import threshold_otsu
import cv2

import utils

def extract_regions(t_img, C_range, R_range):
    """
    Extracts region propsals for a given image
    """
    all_boxes = []    
    for R in R_range:
        for C in C_range:
            s_img = cv2.morphologyEx(t_img, cv2.MORPH_CLOSE, np.ones((R, C), dtype=np.ubyte))
            n, l_img, stats, centroids = cv2.connectedComponentsWithStats(s_img, connectivity=4)
            boxes = [[b[0], b[1], b[0] + b[2], b[1] + b[3]] for b in stats]
            all_boxes += boxes
                
    return all_boxes

    
def find_regions(img, threshold_range, C_range, R_range):
    """
    Finds region proposals in an image using different thresholding techniques and different morphology kernel sizes    
    """

    ims = []
    for t in threshold_range:
        ims.append((img < t).astype(np.ubyte))
    
    ab = []
    for t_img in ims:
        ab += extract_regions(t_img, C_range, R_range)
        
    return ab

def generate_region_proposals(f):
    img = imread(f)
    if img.ndim == 3:
        img = img_as_ubyte(rgb2gray(img))
        
    m = img.mean()
    threshold_range = np.arange(0.6, 1.01, 0.1) * m
    C_range=range(3, 50, 2) #horizontal range
    R_range=range(3, 50, 2) #vertical range
    region_proposals = find_regions(img, threshold_range, C_range, R_range) 
    region_proposals, _ = utils.unique_boxes(region_proposals)
    return region_proposals

def aug_crop(img, tparams):
    t_img = img < threshold_otsu(img)
    nz = t_img.nonzero()
    pad = np.random.randint(low = tparams['hpad'][0], high = tparams['hpad'][1], size=2)    
    vpad = np.random.randint(low = tparams['vpad'][0], high = tparams['vpad'][1], size=2)    
    b = [max(nz[1].min() - pad[0], 0), max(nz[0].min() - vpad[0], 0), 
         min(nz[1].max() + pad[1], img.shape[1]), min(nz[0].max() + vpad[1], img.shape[0])]
    return img[b[1]:b[3], b[0]:b[2]]
    
def affine(img, tparams):
    phi = (np.random.uniform(tparams['shear'][0], tparams['shear'][1])/180) * np.pi
    theta = (np.random.uniform(tparams['rotate'][0], tparams['rotate'][1])/180) * np.pi
    t = tf.AffineTransform(shear=phi, rotation=theta, translation=(-25, -50))
    tmp = tf.warp(img, t, order=tparams['order'], mode='edge', output_shape=(img.shape[0] + 100, img.shape[1] + 100))
    return tmp

def morph(img, tparams):
    ops = [mor.grey.erosion, mor.grey.dilation]
    t = ops[np.random.randint(2)] 
    if t == 0:    
        selem = mor.square(np.random.randint(1, tparams['selem_size'][0]))
    else:
        selem = mor.square(np.random.randint(1, tparams['selem_size'][1]))
    return t(img, selem)    

def augment(word, tparams, keep_size=False):
    assert(word.ndim == 2)
    t = np.zeros_like(word)
    s = np.array(word.shape) - 1
    t[0, :] = word[0, :]
    t[:, 0] = word[:, 0]
    t[s[0], :] = word[s[0], :]
    t[:, s[1]] = word[:, s[1]]
    pad = np.median(t[t > 0])

    tmp = np.ones((word.shape[0] + 8, word.shape[1] + 8), dtype = word.dtype) * pad
    tmp[4:-4, 4:-4] = word
    out = tmp
    out = affine(out, tparams)
    out = aug_crop(out, tparams)        
    out = morph(out, tparams)
    if keep_size:
        out = tf.resize(out, word.shape)
    out = np.round(out).astype(np.ubyte)
    return out

def replace_tokens(text, tokens):
    for t in tokens:
        text = text.replace(t, '')
        
    return text
    
def outer_box(boxes):
    """
    Returns the bounding box of an (Nx4) array of boxes on form [x1, y1, x2, y2]
    """
    return np.array([boxes[:, 0].min(), boxes[:, 1].min(), boxes[:, 2].max(), boxes[:, 3].max()])

def close_crop_box(img, box):
    gray = rgb2gray(img[box[1]:box[3], box[0]:box[2]])
    t_img = gray < threshold_otsu(gray)
    v_proj = t_img.sum(axis=1)
    h_proj = t_img.sum(axis=0)
    y1o = box[1] + max(v_proj.nonzero()[0][0] - 1, 0)
    x1o = box[0] + max(h_proj.nonzero()[0][0] - 1, 0)
    y2o = box[3] - max(v_proj.shape[0] - v_proj.nonzero()[0].max() - 1, 0)
    x2o = box[2] - max(h_proj.shape[0] - h_proj.nonzero()[0].max() - 1, 0)
    obox = (x1o, y1o, x2o, y2o)
    return obox

def fullpage_augment(data, outdir, num_images=2500, augment=True, reset=False):
    output_json = os.path.join(outdir, 'fullpage_augment/data.json')
    if not os.path.exists(output_json) or reset:
        train_data = [datum for datum in data if datum['split'] == 'train']
        vocab, _ = utils.build_vocab(train_data) #vocab local to this function
        vocab_size = len(vocab)
        wtoi = {w:i for i, w in enumerate(vocab)}
                       
        od = osp.join(outdir, 'fullpage_augment')
        if not osp.exists(od):
            os.makedirs(od)
        
        tparams = {}
        tparams['shear'] = (-5, 30)
        tparams['order'] = 1            #bilinear
        tparams['selem_size'] = (3, 4)  #max size for square selem for erosion, dilation
        tparams['rotate'] = (0, 1)
        tparams['hpad'] = (0, 12)
        tparams['vpad'] = (0, 12)
        
        words_by_label = [[] for i in range(vocab_size)] 
        shapes = []
        medians = []
        for datum in train_data:
            img = imread(datum['id'])
            if img.ndim == 3:
                img = img_as_ubyte(rgb2gray(img))
                
            medians.append(np.median(img))
            shapes.append(img.shape)
            for r in datum['regions']:
                x1, y1, x2, y2 = r['x'], r['y'], r['x'] + r['width'], r['y'] + r['height']
                word = img[y1:y2, x1:x2]
                label = r['label']
                ind = wtoi[label]
                words_by_label[ind].append(word)
        
        m = int(np.median(medians))
        new_data = []
        nwords = 256 #batch size?
        s = 3 #inter word space
        box_id = 0
        for i in range(num_images):
            x, y = s, s #Upper left corner of box
            gt_boxes = []
            gt_labels = []
            shape = shapes[i % len(shapes)]
            canvas = create_background(m + np.random.randint(0, 20) - 10, shape)
            maxy = 0
            f = os.path.join(od, '%d.png' % i)
            regions = []
            for j in range(nwords):
                ind = np.random.randint(vocab_size) 
                k = len(words_by_label[ind])
                word = words_by_label[ind][np.random.randint(k)]
                #randomly transform word and place on canvas
                if augment:
                    try:
                        tword = augment(word, tparams)
                    except:
                        tword = word
                else:
                    tword = word
                    
                h, w = tword.shape
                if x + w > shape[1]: #done with row?
                    x = s
                    y = maxy + s
                    
                if y + h > shape[0]: #done with page?
                    break
                
                x1, y1, x2, y2 = x, y, x + w, y + h
                canvas[y1:y2, x1:x2] = tword
                b = [x1, y1, x2, y2]
                gt_labels.append(vocab[ind])
                gt_boxes.append(b)
                x = x2 + s
                maxy = max(maxy, y2)
                
                r = {}
                r['id'] = box_id
                r['image'] = f
                r['height'] = b[3] - b[1]
                r['width'] = b[2] - b[0]
                r['label'] = vocab[ind]
                r['x'] = b[0]
                r['y'] = b[1]
                box_id += 1
                regions.append(r)

            imsave(f, canvas)
            d = {}
            d['gt_boxes'] = gt_boxes
            d['id'] = f
            d['split'] = 'train'
            d['regions'] = regions
            d['region_proposals'] = gt_boxes #Dummy values, not used for when training
            new_data.append(d)
           
        with open(output_json, 'w') as f:
            json.dump(new_data, f)
            
    else:
        with open(output_json) as f:
            new_data = json.load(f)
            
    return new_data
            
def create_background(m, shape, fstd=2, bstd=10):
    canvas = np.ones(shape) * m
    noise = np.random.randn(shape[0], shape[1]) * bstd
    noise = fi.gaussian(noise, fstd)     #low-pass filter noise
    canvas += noise
    canvas = np.round(canvas).astype(np.uint8)
    return canvas
   
def inplace_augment(data, outdir, fold=1, tparams=None, reset=False):
    output_json = osp.join(outdir, 'inplace_augment/data.json')
    if not os.path.exists(output_json) or reset:
        
        od = osp.join(outdir, 'inplace_augment')
        if not osp.exists(od):
            os.makedirs(od)
        
        if tparams == None:
            tparams = {}
            tparams['samples_per_image'] = 5
            tparams['shear'] = (-5, 30)
            tparams['order'] = 1            #bilinear
            tparams['selem_size'] = (3, 4)  #max size for square selem for erosion, dilation
            
        tparams['rotate'] = (0, 1)
        tparams['hpad'] = (0, 1)
        tparams['vpad'] = (0, 1)
            
        augmented = []
        for datum in data:
            dat = copy.deepcopy(datum)
            augmented.append(dat)

            if datum['split'] == 'train':
                datum['region_proposals'] = datum['gt_boxes'][:2] #smaller memory footprint, needed
                path, f = osp.split(datum['id'])
                for i in range(tparams['samples_per_image']):
                    img = imread(datum['id'])
                    if img.ndim == 3:
                        img = img_as_ubyte(rgb2gray(img))
                        
                    out = img.copy()
                    boxes = datum['gt_boxes']
                    for jj, b in enumerate(reversed(boxes)):
                        try: #Some random values for weird boxes give value errors, just handle and ignore
                            b = close_crop_box(img, b)
                            word = img[b[1]:b[3], b[0]:b[2]]
                            aug = augment(word, tparams, keep_size=True)
                        except ValueError:
                            continue
                            
                        out[b[1]:b[3], b[0]:b[2]] = aug
                    
                    new_path = osp.join(od, f[:-4] + '_%d.png' % i)
                    imsave(new_path, out)
                    new_datum = copy.deepcopy(datum)
                    new_datum['id'] = new_path
                    augmented.append(new_datum)
                
        with open(output_json, 'w') as f:
            json.dump(augmented, f)
    
    else: #otherwise load the json
        with open(output_json) as f:
            augmented = json.load(f) 
            
    return augmented
    
def load_washington(fold=1, root="data/washington/"):
    output_json = root + 'washington_fold_%d.json' % fold

    if not os.path.exists(output_json):
        files = sorted(glob.glob(os.path.join(root, 'gw_20p_wannot/*.tif')))
        gt_files = [f[:-4] + '_boxes.txt' for f in files]
        
        with open(os.path.join(root, 'gw_20p_wannot/annotations.txt')) as f:
            lines = f.readlines()

        keep = string.ascii_lowercase + string.digits    
        texts = [l[:-1] for l in lines]        
        ntexts = [replace_tokens(text.lower(), [t for t in text if t not in keep]) for text in texts]
        
        data = []
        ind = 0
        box_id = 0
        for i, (f, gtf) in enumerate(zip(files, gt_files)):
            with open(gtf, 'r') as ff:
                boxlines = ff.readlines()[1:]

            img = imread(f)
            h, w = img.shape
            gt_boxes = []
            for line in boxlines:
                tmp = line.split()
                x1 = int(float(tmp[0]) * w)
                x2 = int(float(tmp[1]) * w)
                y1 = int(float(tmp[2]) * h)
                y2 = int(float(tmp[3]) * h)
                box = (x1, y1, x2, y2)
                gt_boxes.append(box)
                
            labels = ntexts[ind:ind + len(gt_boxes)]
            ind += len(gt_boxes)
            labels = [unicode(l, errors='replace') for l in labels]

            regions = []
            for p, l in zip(gt_boxes, labels):
                r = {}
                r['id'] = box_id
                r['image'] = f
                r['height'] = p[3] - p[1]
                r['width'] = p[2] - p[0]
                r['label'] = l
                r['x'] = p[0]
                r['y'] = p[1]
                regions.append(r)
                box_id += 1

            proposals = generate_region_proposals(f)
            
            datum = {}
            datum['id'] = f
            datum['gt_boxes'] = gt_boxes
            datum['regions'] = regions
            datum['region_proposals'] = proposals.tolist()
            data.append(datum)
            
        if os.path.exists(root + 'washington_crossval_indeces.npz'):
            inds = np.squeeze(np.load(root + 'washington_crossval_indeces.npz')['inds'])
        else:
            inds = np.random.permutation(len(files))
            np.savez_compressed(root + 'washington_crossval_indeces.npz', inds=inds)
        
        data = [data[i] for i in inds]
        
        #Train/val/test on different partitions based on which fold we're using
        data = np.roll(data, 5 * (fold - 1)).tolist()

        for j, datum in enumerate(data):
            if j < 14:
                datum['split'] = 'train'
            elif j == 14:
                datum['split'] = 'val'
            else:
                datum['split'] = 'test'
         
        with open(output_json, 'w') as f:
            json.dump(data, f)
    
    else: #otherwise load the json
        with open(output_json) as f:
            data = json.load(f)
    
    return data  

def filter_ground_truth_boxes(data, image_size=1720):
    """
    Remove too small ground truth boxes when downsampled to the roi-pooling size
    First it's the image scaling preprocessing then it's the downsampling in 
    the network.
    """
    for i, datum in enumerate(data):
        img = imread(datum['id'])
        H, W = img.shape
        scale = float(image_size) / max(H, W)
        
        #Since we downsample the image 8 times before the roi-pooling, 
        #divide scaling by 8. Hardcoded per network architecture.
        scale /= 8
        okay = []
        okay_gt = []
        assert(len(datum['regions']) == len(datum['gt_boxes']))
        for r, gt in zip(datum['regions'], datum['gt_boxes']):
            
            x, y, w, h = r['x'], r['y'], r['width'], r['height']
            xb, yb, wb, hb = gt[0], gt[1], gt[2] - gt[0], gt[3] - gt[1]
            assert(xb == x)
            assert(yb == y)
            assert(wb == w)
            assert(hb == h)
            x, y = round(scale*(x-1)+1), round(scale*(y-1)+1)
            w, h = round(scale*w), round(scale*h)  
            
            if w > 1 and h > 1 and x > 0 and y > 0:
                okay.append(r)
                okay_gt.append(gt)
                
        datum['regions'] = okay
        datum['gt_boxes'] = okay_gt
