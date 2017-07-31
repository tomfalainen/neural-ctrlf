#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 18:58:14 2016

@author: tomas
"""
from multiprocessing import Pool
import json
import string

import scipy.sparse as spa
import h5py
import numpy as np
np.errstate(divide='ignore', invalid='ignore')
from scipy.spatial import distance as di
from skimage.io import imread
import cv2

import dataset_loader as dl
from embeddings import dct
   
def overlap(b1, b2):
    """
    Boxes should be quintuples (x1, y1, x2, y2) defining the upper left and bottom right corners of a box.
    """
    x1 = max(b1[0], b2[0])
    x2 = min(b1[2], b2[2])
    y1 = max(b1[1], b2[1])
    y2 = min(b1[3], b2[3])
    
    ai = max(0, x2 - x1) * max(0, y2 - y1)
    if not ai:
        return 0.0
    
    #calculate box area
    a1 = (b1[2]- b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2]- b2[0]) * (b2[3] - b2[1])

    #find area of intersection
    ai = max(0, min(b1[2], b2[2]) - max(b1[0], b2[0])) * max(0, min(b1[3], b2[3]) - max(b1[1], b2[1]))
        
    if not ai:
        return 0.0
        
    #calculate union    
    au = a1 + a2 - ai
    
    if not au:
        return 0.0
        
    return float(ai) / float(au)

def calculate_overlap(ab, tb):
    overlaps = np.array([[overlap(b1, b2) for b2 in tb] for b1 in ab])
    return overlaps

def extract_regions(t_img, C_range, R_range):
    """
    Extracts region propsals for a given image
    """
    params = []
    all_boxes = []    
    for R in R_range:
        for C in C_range:
            s_img = cv2.morphologyEx(t_img, cv2.MORPH_CLOSE, np.ones((R, C), dtype=np.ubyte))
            n, l_img, stats, centroids = cv2.connectedComponentsWithStats(s_img, connectivity=4)
            boxes = [[b[0], b[1], b[0] + b[2], b[1] + b[3]] for b in stats]
            all_boxes += boxes
            params += [(R, C)] * len(boxes)
                
    return all_boxes, params

def find_regions(img, threshold_range, C_range, R_range):
    """
    Finds region proposals in an image using different thresholding techniques and different morphology kernel sizes    
    """

    ims = []
    for t in threshold_range:
        ims.append((img < t).astype(np.ubyte))
    
    ab = []
    params = []
    for t_img in ims:
        a, p = extract_regions(t_img, C_range, R_range)
        ab += a
        params += p
        
    return ab, params
    
def pad_proposals(proposals, im_shape, pad=10):
    props = []
    for p in proposals:
        pp = [max(0, p[0] - pad), max(0, p[1] - pad), min(im_shape[1], p[2] + pad), min(im_shape[0], p[3] + pad)]
        props.append(pp)
    return np.array(props)
    
def find_matches(ab, tb, overlap_threshold=0.5):
    labels = np.ones((len(ab))) * -1
    
    overlaps = calculate_overlap(ab, tb)
        
    labels = overlaps.max(axis=1)
    inds = overlaps.argmax(axis=1)

    i = labels >= overlap_threshold
    covered = np.unique(inds[i])
    recall = float(len(covered)) / float(len(tb))
    missed = [t for j, t in enumerate(tb) if j not in covered]
    
    return labels, recall, missed, covered

def nms_detections(dets, overlap=0.3):
    """
    Non-maximum suppression: Greedily select high-scoring detections and
    skip detections that are significantly covered by a previously
    selected detection.

    This version is translated from Matlab code by Tomasz Malisiewicz,
    who sped up Pedro Felzenszwalb's code.

    Parameters
    ----------
    dets: ndarray
        each row is ['xmin', 'ymin', 'xmax', 'ymax', 'score']
    overlap: float
        minimum overlap ratio (0.3 default)

    Output
    ------
    dets: ndarray
        remaining after suppression.
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    ind = np.argsort(dets[:, 4])

    w = x2 - x1
    h = y2 - y1
    area = (w * h).astype(float)

    pick = []
    while len(ind) > 1:
        i = ind[-1]         #get highest scoring detection
        pick.append(i)      
        ind = ind[:-1]      #and remove it from detection list

        #calculate overlap 
        xx1 = np.maximum(x1[i], x1[ind])
        yy1 = np.maximum(y1[i], y1[ind])
        xx2 = np.minimum(x2[i], x2[ind])
        yy2 = np.minimum(y2[i], y2[ind])

        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)

        wh = w * h
        if (area[i] + area[ind] - wh).any() == 0:
            continue
            
        o = wh / (area[i] + area[ind] - wh)
                                        
        #remove the detection that overlap too much with highest scoring detection
        ind = ind[np.nonzero(o <= overlap)[0]]
    
    pick = pick[::-1]

    return dets[pick, :], pick

def unique_boxes(boxes):
    tmp = np.array(boxes)
    ncols = tmp.shape[1]
    dtype = tmp.dtype.descr * ncols
    struct = tmp.view(dtype)
    uniq, index = np.unique(struct, return_index=True)
    tmp = uniq.view(tmp.dtype).reshape(-1, ncols)
    return tmp, index
  
def evaluate(args):
    dataset = args.dataset
    root_dir = args.root_dir
    fold = args.fold
    score_nms_overlap = args.score_nms_overlap #For wordness scores from network
    score_threshold = args.score_threshold
    nms_overlap = args.nms_overlap
    overlap_threshold = args.overlap_threshold
    num_queries = args.num_queries
    split = args.split
    embedding = args.embedding
    
    with open(root_dir + '%s_fold%d.json' % (dataset, fold), 'r') as f:
        jdata = json.load(f)
    
    wtoi = jdata['wtoi']

    f = h5py.File(root_dir + '%s_fold%d.h5' % (dataset, fold), 'r')
    heights = f['image_heights'].value
    widths = f['image_widths'].value
    original_heights = f['original_heights'].value
    original_widths = f['original_widths'].value
    f.close()
    
    data = getattr(dl, 'load_' + dataset)(fold)    
    vocab, indeces = build_vocab(data)
    alphabet = string.ascii_lowercase + string.digits
    wtoe = {w:dct(w, 3, alphabet) for w in vocab}
    
    if split == 'test' or split == 'train':
        desc_dir = 'descriptors/'
    elif split == 'val':
        desc_dir = 'tmp/'    
    else:
        raise ValueError("Invalid split option")
    #nms on wordness scores done in torch
    f = h5py.File(desc_dir +  '%s_%s_descriptors.h5' % (dataset, embedding), 'r')
    all_gt_boxes = f['gt_boxes_fold%d' % fold].value.astype(np.int32)
    gt_embeddings = f['gt_embeddings_fold%d' % fold].value
    rpn_embeddings = f['embeddings_fold%d' % fold].value
    proposal_embeddings = f['rp_embeddings_fold%d' % fold].value
    proposal_logprobs = f['rp_scores_fold%d' % fold].value
    all_rps = f['region_proposals_fold%d' % fold].value.astype(np.int32)
    logprobs = np.squeeze(f['logprobs_fold%d' % fold].value)
    all_boxes = f['boxes_fold%d' % fold].value.astype(np.int32)
    btoi = f['box_to_images_fold%d' % fold].value.astype(np.int32)
    rptoi = f['rp_to_images_fold%d' % fold].value.astype(np.int32)
    f.close()
    
    qbe_queries = []
    qbs_queries = []
    qbs_qtargets = []
    qbe_qtargets = []
    db_targets = []
    gt_targets = []
    
    all_embeddings = []
    joint_boxes = []
    overlaps = []
    used = []
    offset = [0, 0]
    ind = 1 #index for test data
    for i, datum in enumerate(data):
        if datum['split'] == split:
            regions = datum['regions']
            gt_boxes = np.array(datum['gt_boxes'])
            gt_labels = [r['label'] for r in regions]
            datum['gt_targets'] = np.array([wtoi[w] for w in gt_labels])
            gt_targets += [wtoi[w] for w in gt_labels]
            
            #Setup qbs queries
            for r in regions:
                label = r['label']
                if label not in used:
                    qbs_queries.append(wtoe[label])
                    qbs_qtargets.append(wtoi[label])
                    used.append(label)
                
            proposals = all_boxes[btoi == ind]
            scores = 1 / (1 + np.exp(-logprobs[btoi == ind][:, np.newaxis])) #convert to probabilities
            embeddings = rpn_embeddings[btoi == ind]
                                  
            if args.use_external_proposals: 
                pp = all_rps[rptoi == ind]
                
                proposals = np.vstack((proposals, pp)) 
                pscores = 1 / (1 + np.exp(-proposal_logprobs[rptoi == ind])) #convert to probabilities
                pemb = proposal_embeddings[rptoi == ind]
                embeddings = np.vstack((embeddings, pemb))
                scores = np.vstack((scores, pscores))
                
            #Scale boxes back to original size
            sh, sw = heights[i], widths[i]
            img_shape = np.array([original_heights[i], original_widths[i]])
            scale = float(max(img_shape)) / max(sh, sw)
            proposals = np.round((proposals -1) * scale + 1).astype(np.int32)
            
            if args.use_external_proposals:
                nrpn = np.sum(btoi==ind)
                rpn_proposals = proposals[:nrpn]
                rp_proposals = proposals[nrpn:]
    
            threshold_pick = np.squeeze(scores > score_threshold)
            scores = scores[threshold_pick]
            proposals = proposals[threshold_pick, :]
            embeddings = embeddings[threshold_pick, :]

            #calculate the different recalls before NMS
            _, recall, missed, covered = find_matches(proposals, gt_boxes, overlap_threshold)
            datum['pre_nms_recall'] = recall

            if args.use_external_proposals:
                rpn_proposals = rpn_proposals[threshold_pick[:nrpn]]
                rp_proposals = rp_proposals[threshold_pick[nrpn:]]

                try:
                    _, rp_recall, __, ___ = find_matches(rp_proposals, gt_boxes, overlap_threshold)
                except AttributeError:
                    rp_recall = 0
                datum['pre_nms_rp_recall'] = rp_recall

                try:        
                    _, rpn_recall, __, ___ = find_matches(rpn_proposals, gt_boxes, overlap_threshold)
                except AttributeError:
                    rpn_recall = 0
                datum['pre_nms_rpn_recall'] = rpn_recall


            dets = np.hstack((proposals, scores))   
            _, pick = nms_detections(dets, overlap=score_nms_overlap)
            tt = np.zeros(len(dets), dtype=np.bool)
            tt[pick] = 1 
    
            proposals = proposals[tt, :]
            embeddings = embeddings[tt, :]
            scores = scores[tt]

            if args.use_external_proposals:
                nrpn = rpn_proposals.shape[0]
                rpn_proposals = rpn_proposals[tt[:nrpn]]
                rp_proposals = rp_proposals[tt[nrpn:]]
    
            all_embeddings.append(embeddings)
            
            #calculate the different recalls
            _, recall, missed, covered = find_matches(proposals, gt_boxes, overlap_threshold)
            datum['recall'] = recall

            if args.use_external_proposals:
                try:
                    _, rp_recall, __, ___ = find_matches(rp_proposals, gt_boxes, overlap_threshold)
                except AttributeError:
                    rp_recall = 0
                datum['rp_recall'] = rp_recall

                try:        
                    _, rpn_recall, __, ___ = find_matches(rpn_proposals, gt_boxes, overlap_threshold)
                except AttributeError:
                    rpn_recall = 0
                datum['rpn_recall'] = rpn_recall
    
            # Artificially make a huge image containing all the boxes to be able to
            # perform nms on distance to query
            for b in proposals:
                bb = [b[0] + offset[1], b[1] + offset[0], b[2] + offset[1], b[3] + offset[0]]
                joint_boxes.append(bb)
        
            offset[0] += img_shape[0]
            offset[1] += img_shape[1]
            
            overlap = calculate_overlap(proposals, gt_boxes)
            overlaps.append(overlap)
            #Sometimes proposals will overlap 0% with any ground truth box, how to handle this?
            inds = overlap.argmax(axis=1)
            proposal_labels = [gt_labels[i] for i in inds]
            datum['proposals'] = proposals
            datum['proposal_labels'] = proposal_labels
            datum['proposal_targets'] = np.array([wtoi[w] for w in proposal_labels])
            datum['overlap'] = overlap
        
            #Integer labels for queries
            qbe_qtargets += datum['gt_targets'].tolist()
            
            #Integer labels for database
            db_targets += datum['proposal_targets'].tolist()
            ind += 1
            
    db = np.vstack(all_embeddings)
    
    all_overlaps = np.zeros((len(joint_boxes), len(all_gt_boxes)), dtype=np.float32)
    x, y = 0, 0
    for o in overlaps:
        all_overlaps[y:y + o.shape[0], x: x + o.shape[1]] = o
        y += o.shape[0]    
        x += o.shape[1]

    db_targets = np.array(db_targets)
    qbe_qtargets = np.array(qbe_qtargets)
    qbs_qtargets = np.array(qbs_qtargets)
    qbe_queries = gt_embeddings
    qbs_queries = np.array(qbs_queries)
    joint_boxes = np.array(joint_boxes)
    
    assert(qbe_queries.shape[0] == qbe_qtargets.shape[0])
    assert(qbs_queries.shape[0] == qbs_qtargets.shape[0])
    assert(db.shape[0] == db_targets.shape[0])
    
    total_recall = np.mean([datum['pre_nms_recall'] for datum in data if datum['split'] == split])
    if args.use_external_proposals:
        rpn_recall = np.mean([datum['pre_nms_rpn_recall'] for datum in data if datum['split'] == split])
        rp_recall = np.mean([datum['pre_nms_rp_recall'] for datum in data if datum['split'] == split])
    else:
        rpn_recall = total_recall
        rp_recall = -1

    if num_queries < 1:
        num_queries = len(qbe_queries) + len(qbs_queries) + 1

    qbe_queries = qbe_queries[:num_queries]
    qbs_queries = qbs_queries[:num_queries]
    
    num_workers = 8        
    mAP_qbe, mR_qbe = mAP_parallel(qbe_queries, qbe_qtargets, db, db_targets,
                                      gt_targets, joint_boxes, all_overlaps, 
                                      nms_overlap, overlap_threshold, num_workers)
    mAP_qbs, mR_qbs = mAP_parallel(qbs_queries, qbs_qtargets, db, db_targets,
                                      gt_targets, joint_boxes, all_overlaps, 
                                  nms_overlap, overlap_threshold, num_workers)
    
    return mAP_qbe, mR_qbe, mAP_qbs, mR_qbs, total_recall, rpn_recall, rp_recall

def average_precision_segfree(res, t, o, sinds, n_relevant, ot):
    """
    Computes the average precision
    
    res: sorted list of labels of the proposals, aka the results of a query.
    t: transcription of the query
    o: overlap matrix between the proposals and gt_boxes.
    sinds: The gt_box with which the proposals overlaps the most.
    n_relevant: The number of relevant retrievals in ground truth dataset
    ot: overlap_threshold
    """
    correct_label = res == t
    
    #The highest overlap between a proposal and a ground truth box
    tmp = []
    covered = []
    for i in range(len(res)):
        if sinds[i] not in covered: #if a ground truth box has been covered, mark proposal as irrelevant
            tmp.append(o[i, sinds[i]])
            if o[i, sinds[i]] >= ot and correct_label[i]:
                covered.append(sinds[i])
        else:
            tmp.append(0.0)
    
    covered = np.array(covered)
    tmp = np.array(tmp)
    relevance = correct_label * (tmp >= ot)
    rel_cumsum = np.cumsum(relevance, dtype=float)
    precision = rel_cumsum / np.arange(1, relevance.size + 1)

    if n_relevant > 0:
        ap = (precision * relevance).sum() / n_relevant
    else:
        ap = 0.0
        
    return ap, covered
    
def hh(arg):
    query, t, db, db_targets, metric, joint_boxes, nms_overlap, all_overlaps, inds, gt_targets, ot, num_workers = arg
    count = np.sum(db_targets == t)
    if count == 0: #i.e., we have missed this word completely
        return 0.0, 0.0
    
    dists = np.squeeze(di.cdist(query[np.newaxis, :], db, metric=metric))   
    sim = (dists.max()) - dists

    dets = np.hstack((joint_boxes, sim[:, np.newaxis]))
    nmsdets, pick = nms_detections(dets, nms_overlap)

    dists = dists[pick]
    I = np.argsort(dists)
    res = db_targets[pick][I]   #Sort results after distance to query image
    o = all_overlaps[pick][I, :]
    sinds = inds[pick][I]
    n_relevant = np.sum(gt_targets == t)
    ap, covered = average_precision_segfree(res, t, o, sinds, n_relevant, ot)
    r = float(np.unique(covered).shape[0]) / n_relevant
    return ap, r
    
def mAP_parallel(queries, qtargets, db, db_targets, gt_targets, joint_boxes, all_overlaps, nms_overlap, ot, num_workers=8):
    metric='cosine'
    inds = all_overlaps.argmax(axis=1)
    all_overlaps = spa.csr_matrix(all_overlaps)
    args = [(q, t, db, db_targets, metric, joint_boxes, nms_overlap, all_overlaps, inds, gt_targets, ot, num_workers) for q, t in zip(queries, qtargets)]
    if num_workers > 1:
        p = Pool(num_workers)
        res = p.map(hh, args)
    else:
        res = []
        for ii, arg in enumerate(args):
            if ii % 1000 == 0:
                print "iter %d of %d" % (ii, len(args))
            res.append(hh(arg))
        res = [hh(arg) for arg in args]
               
    res = np.array(res)
    return np.mean(res, axis=0)
    
def filter_region_proposals(data, original_heights, original_widths, image_size):
    """
    Remove duplicate region proposals when downsampled to the roi-pooling size
    First it's the image scaling preprocessing then it's the downsampling in 
    the network.
    """
    for i, datum in enumerate(data):
        H, W = original_heights[i], original_widths[i]
        scale = float(image_size) / max(H, W)
        
        #Since we downsample the image 8 times before the roi-pooling, 
        #divide scaling by 8. Hardcoded per network architecture.
        scale /= 8
        okay = []
        for box in datum['region_proposals']:
            x, y = box[0], box[1]
            w, h = box[2] - x, box[3] - y
            x, y = round(scale*(x-1)+1), round(scale*(y-1)+1)
            w, h = round(scale*w), round(scale*h)  
            
            if x < 1: x = 1
            if y < 1: y = 1

            if w > 0 and h > 0:
                okay.append(box)
            
        #Only keep unique proposals in downsampled coordinate system, i.e., remove aliases 
        region_proposals, _ = unique_boxes(np.array(okay))
        datum['region_proposals'] = region_proposals

def build_vocab(data):
    """ Builds a set that contains the vocab. Filters infrequent tokens. """
    texts = []
    for datum in data:
        for r in datum['regions']:
            texts.append(r['label'])
  
    vocab, indeces = np.unique(texts, return_index=True)
    return vocab, indeces
