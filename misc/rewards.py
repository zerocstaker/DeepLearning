from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time
import misc.utils as utils
from collections import OrderedDict
import torch

import sys
sys.path.append("cider")
from pyciderevalcap.ciderD.ciderD import CiderD
sys.path.append("coco-caption")
from pycocoevalcap.bleu.bleu import Bleu

CiderD_scorer = None
Bleu_scorer = None
#CiderD_scorer = CiderD(df='corpus')

def init_scorer(cached_tokens):
    global CiderD_scorer
    CiderD_scorer = CiderD_scorer or CiderD(df=cached_tokens)
    global Bleu_scorer
    Bleu_scorer = Bleu_scorer or Bleu(4)

def array_to_str(arr):
    out = ''
    for i in range(len(arr)):
        out += str(arr[i]) + ' '
        if arr[i] == 0:
            break
    return out.strip()

def get_self_critical_reward(model, fc_feats, att_feats, att_masks, data_gts, gen_result_list, opt):
    batch_size = gen_result_list[0].size(0)# batch_size = sample_size * seq_per_img
    length = gen_result_list[0].shape[1]
    seq_per_img = batch_size // len(data_gts)
    
    # get greedy decoding baseline
    # model.eval()
    # with torch.no_grad():
    #     greedy_res, _ = model(fc_feats, att_feats, att_masks=att_masks, mode='sample')
    # model.train()
    scores_list = []
    for gen_result in gen_result_list:
        res = OrderedDict()
        
        gen_result = gen_result.data.cpu().numpy()

        #We just need the result part
        for i in range(batch_size):
            res[i] = [array_to_str(gen_result[i])]

        gts = OrderedDict()
        for i in range(len(data_gts)):
            gts[i] = [array_to_str(data_gts[i][j]) for j in range(len(data_gts[i]))]

        #looks like this is just calculating the raw score for all of the words
        res_ = [{'image_id':i, 'caption': res[i]} for i in range(batch_size)]
        res__ = {i: res[i] for i in range(batch_size)}
        gts = {i: gts[i % batch_size // seq_per_img] for i in range(batch_size)}
        if opt.cider_reward_weight > 0:
            _, cider_scores = CiderD_scorer.compute_score(gts, res_)
            print('Cider scores:', _)
        else:
            cider_scores = 0
        if opt.bleu_reward_weight > 0:
            _, bleu_scores = Bleu_scorer.compute_score(gts, res__)
            bleu_scores = np.array(bleu_scores[3])
            print('Bleu scores:', _[3])
        else:
            bleu_scores = 0
        scores = opt.cider_reward_weight * cider_scores + opt.bleu_reward_weight * bleu_scores

        #Do the deduction here, so this is really the update value
        scores_list.append(scores)

    rewards = np.zeros(batch_size, length)
    #calculate the reward for each word
    for i in range(len(scores_list) - 1):
        #should do element wise deduction here
        rewards[:,i] = numpy.subtract(scores_list[i], scores_list[i+1])

    return rewards