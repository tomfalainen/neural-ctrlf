#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 22:50:52 2017

@author: tomas
"""

import os
import string
import argparse
from subprocess import Popen, PIPE

from matplotlib.colors import LinearSegmentedColormap as lsc
import h5py
import numpy as np
from scipy.misc import imresize
np.errstate(divide='ignore', invalid='ignore')
from scipy.spatial import distance as di
from skimage.io import imread, imsave
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte
from skimage import draw

import misc.dataset_loader as dl
from misc.embeddings import dct
from misc.utils import nms_detections, pad_proposals

def filter_region_proposals(region_proposals, H0, W0, image_size):
    """
    Remove duplicate region proposals when downsampled to the roi-pooling size
    First it's the image scaling preprocessing then it's the downsampling in 
    the network.
    """
    
    scale = float(image_size) / max(H0, W0)
    
    #Since we downsample the image 8 times before the roi-pooling, 
    #divide scaling by 8. Hardcoded per network architecture.
    scale /= 8
    
    okay = []
    for box in region_proposals:
        x, y = box[0], box[1]
        w, h = box[2] - x, box[3] - y
        x, y = round(scale*(x-1)+1), round(scale*(y-1)+1)
        w, h = round(scale*w), round(scale*h)  
        
        if x < 1: x = 1
        if y < 1: y = 1

        if w > 0 and h > 0:
            okay.append(box)
        
    return np.array(okay)

def extract_features(args, img, region_proposals):
    if not os.path.exists('tmp/'):
        os.makedirs('tmp/')
        
    washington_mean = 60.7622885
    img = img.astype(np.float32) - washington_mean
    img = img[np.newaxis, np.newaxis, :, :]        
        
    #Write the data
    h5_file  = 'tmp/single_image.h5'
    f = h5py.File(h5_file, 'w')
    f.create_dataset('/img', data=img)
    f.create_dataset('/region_proposals', data=region_proposals)
    f.close()
    
    tmp = (args.model, h5_file, args.gpu)
    command = """th single_image.lua -model_file %s -h5_file %s -gpu %d""" % tmp
    Popen(command, shell=True, stdout=PIPE).stdout.read()

    #Read the results
    f = h5py.File(h5_file, 'r')
    boxes = f['boxes'].value
    embeddings = f['embeddings'].value
    logprobs = f['logprobs'].value
    rp_embeddings = f['rp_embeddings'].value
    rp_scores = f['rp_scores'].value
    f.close()

    ret = (boxes, embeddings, logprobs, rp_embeddings, rp_scores)
    return ret 

def draw_boxes_onto_image(img, ab, colors=(1, 0, 0), lw=3):
    h, w = img.shape
    oshape = (h + 2*lw, w + 2*lw, 3)
    out = np.zeros(oshape, dtype=img.dtype)    
    if img.ndim == 2:
        out[lw:-lw, lw:-lw, 0] = img 
        out[lw:-lw, lw:-lw, 1] = img 
        out[lw:-lw, lw:-lw, 2] = img
    else:
        out[lw:-lw, lw:-lw, :] = img.copy()
    
    if type(colors) == tuple:
        colors = [colors] * len(ab)
        
    for b, color in zip(ab, colors):
        color = np.array(list(color)) * 255
        
        b = [b[0] + lw, b[1] + lw, b[2] + lw, b[3] + lw]
        
        #If outside image, don't draw
        if b[0] < 0 or b[1] < 0 or b[2] >= w or b[3] >= h:
#            continue
            if b[3] >= h:
                b[3] = h - 1
            if b[2] >= w:
                b[2] = w - 1
        
        for i in np.arange(-(lw-1)/2, (lw-1)/2 + 1, 1):
            
            #(y1, x1 + i) to (y2, x1 + i)
            rr, cc = draw.line(b[1], b[0] + i, b[3], b[0] + i)
            out[rr, cc, :] = color

            #(y1, x2 + i) to (y2, x2 + i)
            rr, cc = draw.line(b[1], b[2] + i, b[3], b[2] + i)
            out[rr, cc, :] = color
            
            #(y1 + i, x1) to (y1 + i, x2)
            rr, cc = draw.line(b[1] + i, b[0], b[1] + i, b[2])
            out[rr, cc, :] = color
            
            #(y2 + i, x1) to (y2 + i, x2)
            rr, cc = draw.line(b[3] + i, b[0], b[3] + i, b[2])
            out[rr, cc, :] = color
            
    return out

def main(args):
    alphabet = string.ascii_lowercase + string.digits
    dct_resolution = 3
    metric = 'cosine'
    score_threshold = 0.01
    score_nms_overlap = 0.5
    nms_overlap = 0.01
    image_size = 1720
    use_external_proposals = True
    
    img = imread(args.image)
    if img.ndim == 3:
        img = img_as_ubyte(rgb2gray(img))
    H0, W0 = img.shape
    down_scale = float(image_size) / max(H0, W0)
    img = imresize(img, down_scale)
    img = np.invert(img)
    H, W = img.shape
    
    try:
        query = dct(args.query.lower(), dct_resolution, alphabet)
    except:
        raise ValueError("Please ensure that your query is in [a-z0-9]")

    region_proposals_orig = dl.generate_region_proposals(args.image)
    region_proposals = filter_region_proposals(region_proposals_orig, H0, W0, image_size)
    region_proposals = pad_proposals(region_proposals, (H0, W0))
    region_proposals = np.round((region_proposals -1) * down_scale + 1).astype(np.int32)

#    ret = extract_features(args.model, img, region_proposals)
    ret = extract_features(args, img, region_proposals)
    rpn_boxes_final, rpn_embeddings, rpn_logprobs, dtp_embeddings, dtp_logprobs = ret
        
    proposals = rpn_boxes_final
    embeddings = rpn_embeddings
    scores = 1 / (1 + np.exp(-rpn_logprobs[:])) #convert to probabilities
    
    if use_external_proposals:       
        proposals = np.vstack((proposals, region_proposals)) 
        pscores = 1 / (1 + np.exp(-dtp_logprobs)) #convert to probabilities
        embeddings = np.vstack((embeddings, dtp_embeddings))
        scores = np.vstack((scores, pscores))
    
    #Scale boxes back to original size
    up_scale = 1/down_scale
    proposals = np.round((proposals -1) * up_scale + 1).astype(np.int32)
        
    threshold_pick = np.squeeze(scores > score_threshold)
    scores = scores[threshold_pick]
    embeddings = embeddings[threshold_pick, :]
    proposals = proposals[threshold_pick, :]

    dets = np.hstack((proposals, scores))   
    _, pick = nms_detections(dets, overlap=score_nms_overlap)
    tt = np.zeros(len(dets), dtype=np.bool)
    tt[pick] = 1 

    proposals = proposals[tt, :]
    scores = scores[tt]
    embeddings = embeddings[tt, :]

    dists = np.squeeze(di.cdist(query[np.newaxis, :], embeddings, metric=metric))   
    sim = (dists.max()) - dists
    nmsdets, pick = nms_detections(np.hstack((proposals, sim[:, np.newaxis])), nms_overlap)
    dists = dists[pick]
    I = np.argsort(dists)
    dists = dists[I]
    top_proposals = proposals[pick][I][:args.nwords]

    dists = dists[:args.nwords]
    ndists = dists/dists.max()

    l = [(0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
    colormap = lsc.from_list('mymap', l, N=args.nwords, gamma=1.0)
    colors = [colormap(nd)[:-1] for nd in ndists]

    lw = 5
    img = imread(args.image)
    im = draw_boxes_onto_image(img, top_proposals, colors, lw)
    imsave(args.out, im)
                  
if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('--root_dir',
    default='data/dbs/',
    help='Path to where your data is located')
    parser.add_argument('--model',
     default='models/ctrlfnet_washington.t7',
    help='The trained Ctrl-F-Net model to use')
    parser.add_argument('--image',
    default='examples/test_image.png',
#    default='data/washington/gw_20p_wannot/3050305.tif',
    help='The image to test on')
    parser.add_argument('--out',
    default='examples/out.png',
    help='The output image')
    parser.add_argument('--nwords',
    default=20,
    help='How many search results to draw')
    parser.add_argument('--gpu',
    default=0,
    help='Which GPU to use')
    parser.add_argument('--query',
    default='orders',
    help='Your search query')
    args = parser.parse_args()
    main(args)

