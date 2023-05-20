import argparse
import os
from os.path import join
import torch
import multiprocessing
import sys

# lightGCN + sentence_transformer
def parse_args():
    parser = argparse.ArgumentParser(description="aspects for diversity")
    parser.add_argument('--device', type=str, default='cuda:0',
                        help="GPU or CPU")
    parser.add_argument('--A_split', type=bool, default=False)
    parser.add_argument('--bigdata', type=bool, default=False)
    parser.add_argument('--bpr_batch', type=int, default=2048,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--recdim', type=int, default=64,
                        help="the embedding size of lightGCN")
    parser.add_argument('--aspect_LMdim', type=int, default=768,  # 384
                        help="the embedding size of the pre-treated aspect")
    parser.add_argument('--layer', type=int, default=2,
                        help="the layer num of lightGCN")
    parser.add_argument('--lr', type=float, default=0.001,
                        help="the learning rate")
    """parser.add_argument('--regloss1_decay', type=float, default=1e-4,
                        help="the weight decay for l2 normalizaton of user/item weight")"""
    parser.add_argument('--regloss_decay', type=float, default=1e-6,  # best: 1e-6
                        help="the weight decay for l2 normalizaton of the model")
    parser.add_argument('--au_loss', type=float, default=1, help='The weight 𝛾 of 𝑙_uniform')
    parser.add_argument('--dropout', type=int, default=0,
                        help="using the dropout or not")
    parser.add_argument('--keepprob', type=float, default=0.6,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--a_fold', type=int, default=100,
                        help="the fold num used to split large adj matrix, like gowalla")
    parser.add_argument('--testbatch', type=int, default=100,  # 100
                        help="the batch size of users for testing")
    parser.add_argument('--dataset', type=str, default='beauty',
                        help="available datasets: [yelp2018, beauty]")
    parser.add_argument('--path', type=str, default="./checkpoints",
                        help="path to save weights")
    parser.add_argument('--runs_path', type=str, default="./runs",
                        help="path to save results")
    parser.add_argument('--topks', nargs='?', default="[10, 20, 50]",
                        help="@k test list")
    parser.add_argument('--tensorboard', type=int, default=1,
                        help="enable tensorboard")
    parser.add_argument('--comment', type=str, default="lgn")
    parser.add_argument('--load', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--multicore', type=int, default=0, help='whether we use multiprocessing or not in test')
    parser.add_argument('--pretrain', type=int, default=0, help='whether we use pretrained weight or not')  # 0
    parser.add_argument('--seed', type=int, default=2023, help='random seed')
    parser.add_argument('--model', type=str, default='lgn', help='rec-model, support [lgn, ...]')
    parser.add_argument('--max_len', type=int, default=32, help='Maximum length of aspect list')  # 32较好
    parser.add_argument('--head_num', type=int, default=1, help='The number of heads of MultiheadAttention') # have little impact
    parser.add_argument('--gamma', type=int, default=0.5, help='The weight 𝛾 of 𝑙_uniform')  # 0.5 best
    parser.add_argument('--sigma', default = 1.0, type = float, help = 'sigma for gaussian kernel')
    parser.add_argument('--gamma2', default = 2.0, type = float, help = 'gamma for gaussian kernel')
    parser.add_argument('--k', default = 20, type = int, help = 'neighbor number in each GNN aggregation(for diversity)')
    parser.add_argument('--loss_function', default = 'au', type = str, help = 'loss function')
    parser.add_argument('--patience', default = 300, type = int, help = 'early_stop patience')  # 100
    parser.add_argument('--new_aspects', default = 8, type = int, help = 'new aspects add to user')  # 16  maybe 32 is best
    parser.add_argument('--new_λ', default = 0.2, type = int, help = 'the weight of the new aspects add to user')  # new_λ
    # for Contrastive Learning
    parser.add_argument('--eps', default = 0.2, type = int, help = 'the weight noise in XsimGCL')  # 0.2
    parser.add_argument('--perturbed', default = False, type = bool, help = 'Add noise or not')  #
    parser.add_argument('--layer_cl', default = 1, type = int, help = 'Layers for contrast') # layer_cl  1-layer
    parser.add_argument('--cl_rate', default = 0.2, type = int, help = 'the weight of cl_loss') # 0.2  best
    parser.add_argument('--temp', default = 0.15, type = int, help = 'the temperature of cl_loss') # -temp
    return parser.parse_args()

