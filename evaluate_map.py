#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 15:33:43 2016

@author: tomas
"""

import argparse

import numpy as np

from misc.utils import evaluate

def main(args):  
    mAP_qbe05, mR_qbe05, mAP_qbs05, mR_qbs05 = [], [], [], []
    total_recall05, rpn_recall05, rp_recall05 = [], [], []
    mAP_qbe025, mR_qbe025, mAP_qbs025, mR_qbs025 = [], [], [], []
    total_recall025, rpn_recall025, rp_recall025 = [], [], []

    for fold in range(1, args.folds + 1):
        args.fold = fold
        args.overlap_threshold = 0.5
        mAP_qbe, mR_qbe, mAP_qbs, mR_qbs, total_recall, rpn_recall, rp_recall = evaluate(args)
            
        mAP_qbe05.append(mAP_qbe * 100)
        mR_qbe05.append(mR_qbe * 100)
        mAP_qbs05.append(mAP_qbs * 100)
        mR_qbs05.append(mR_qbs * 100)
        total_recall05.append(total_recall * 100)
        rpn_recall05.append(rpn_recall * 100)
        rp_recall05.append(rp_recall * 100)
        
        if args.verbose:
            print ''
            print 'Overlap Threshold: 50%'
            print 100*mAP_qbe, 100*mR_qbe
            print 100*mAP_qbs, 100*mR_qbs
            print 100*total_recall, 100*rpn_recall, 100*rp_recall

        args.overlap_threshold = 0.25
        mAP_qbe, mR_qbe, mAP_qbs, mR_qbs, total_recall, rpn_recall, rp_recall = evaluate(args)
            
        mAP_qbe025.append(mAP_qbe * 100)
        mR_qbe025.append(mR_qbe * 100)
        mAP_qbs025.append(mAP_qbs * 100)
        mR_qbs025.append(mR_qbs * 100)
        total_recall025.append(total_recall * 100)
        rpn_recall025.append(rpn_recall * 100)
        rp_recall025.append(rp_recall * 100)

        if args.verbose:
            print ''
            print 'Overlap Threshold: 25%'
            print 100*mAP_qbe, 100*mR_qbe
            print 100*mAP_qbs, 100*mR_qbs
            print 100*total_recall, 100*rpn_recall, 100*rp_recall

    print ''
    print '%s, embedding = %s' % (args.dataset, args.embedding)
    print 'Overlap Threshold: 50%                                   25%'
    print 'QbE MAP: %f, mR: %f           QbE MAP: %f, mR: %f' % (np.mean(mAP_qbe05), 
                         np.mean(mR_qbe05), np.mean(mAP_qbe025), np.mean(mR_qbe025))
    print 'QbS MAP: %f, mR: %f           QbS MAP: %f, mR: %f' % (np.mean(mAP_qbs05), 
                         np.mean(mR_qbs05), np.mean(mAP_qbs025), np.mean(mR_qbs025))
    print ''
    print 'Overlap 50%% Recall: %f, RPN recall: %f, RP recall: %f' % (np.mean(total_recall05), 
                                                   np.mean(rpn_recall05), np.mean(rp_recall05))
    print 'Overlap 25%% Recall: %f, RPN recall: %f, RP recall: %f' % (np.mean(total_recall025), 
                                                   np.mean(rpn_recall025), np.mean(rp_recall025))

if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('--dataset',
    default='washington',
    help='Which dataset to evaluate.')
    parser.add_argument('--root_dir',
    default='data/dbs/',
    help='Path to where your data is located')
    parser.add_argument('--cross_val',
    default=0,
    help='Whether to use 4-fold cross validation or not')
    parser.add_argument('--num_queries',
    default=-1,
    help='Number of queries to use for evaluation. -1 = all queries')
    parser.add_argument('--num_workers',
    default=1,
    help='Number of workers to use for evaluation.')
    parser.add_argument('--embedding',
    default='dct',
    help='Which embedding to use')
    parser.add_argument('--verbose',
    default=0,
    help='Print individual fold scores or not')
    args = parser.parse_args()
    
    if args.cross_val:
        args.folds = 4
    else:
        args.folds = 1

    #Add some extra parameters
    args.score_nms_overlap = 0.4
    args.score_threshold = 0.01
    args.nms_overlap = 0.01
    args.use_external_proposals = True
    args.split = 'test'  
    main(args)
