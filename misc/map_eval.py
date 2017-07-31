#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 11:41:27 2016

@author: tomas
"""

import sys
import json

from misc.utils import evaluate

if __name__ == '__main__':
    fold = int(sys.argv[1])
    dataset = str(sys.argv[2])
    embedding = str(sys.argv[3])

    root_dir='data/dbs/'   
    args = {}
    args['score_nms_overlap'] = 0.75
    args['score_threshold'] = 0.01
    args['nms_overlap'] = 0.1
    args['overlap_threshold'] = 0.5
    args['use_external_proposals'] = True
    args['split'] = 'val'
    args['embedding'] = embedding
    args['num_queries'] = -1
    args['fold'] = fold
    args['root_dir'] = root_dir
    args['dataset'] = dataset
    
    mAP_qbe, mR_qbe, mAP_qbs, mR_qbs, total_recall, rpn_recall, rp_recall = evaluate(args)

    jdata = {}
    jdata['mAP_qbe'] = mAP_qbe
    jdata['recall_qbe'] = mR_qbe
    jdata['mAP_qbs'] = mAP_qbs
    jdata['recall_qbs'] = mR_qbs
    jdata['total_recall'] = total_recall
    jdata['rpn_recall'] = rpn_recall
    jdata['rp_recall'] = rp_recall

    #store results in a json file
    with open('tmp/' + dataset + '_fold%d_ws_results.json' % fold, 'w') as f:
        json.dump(jdata, f)        
