# RUNNING COMMAND:
# python ColFiltering.py [uu/mm/pcc/mf] [cosine/dot] [mean/weight] [k] [output_filepath]

__author__ = 'luoshalin'

from scipy import sparse
import os
import sys
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine
from scipy import spatial
import heapq
from numpy import linalg as LAn
import time
from scipy.sparse import *
from scipy import *

from preprocess import get_trainM, get_qmM
from user_sim import get_user_user_pred, uu_output
from movie_sim import get_movie_movie_pred, mm_output


def main(argv):
    # TIMER
    t0 = time.time()

    # PARAMETERS
    model_arg = sys.argv[1]
    sim_arg = sys.argv[2]
    weight_arg = sys.argv[3]
    k = int(sys.argv[4])
    dev_filepath = '../../data/HW4_data/test.csv'
    test_filepath = '../../data/HW4_data/test.csv'
    train_filepath = '../../data/HW4_data/train.csv'
    dev_query_filepath = '../../data/HW4_data/dev.queries'
    test_query_filepath = '../../data/HW4_data/test.queries'
    output_filepath = sys.argv[5]

    # DATA PREPROCESSING
    # read train data -> preprocessing into vectors -> imputation
    trainM = get_trainM(train_filepath)  # get a sparse M as training set, <user, movie>[score]
    # save target <query, movie> pairs to be predicted into a matrix
    mu_list = []
    qmM = get_qmM(dev_filepath, mu_list)


    # ==========/ EXP 1(uu) /========== #
    if model_arg == 'uu':
        qid_set = set(find(qmM)[0])  # row - queries
        qid_list = list(qid_set)
        # run user-user similarity algo
        uu_pred_dic = get_user_user_pred(qid_list, trainM, k, sim_arg, weight_arg)
        # write to result
        uu_output(uu_pred_dic, mu_list, dev_filepath, output_filepath)

    # ==========/ EXP 2(mm) /========== #
    if model_arg == 'mm':
        # run movie-movie similarity algo
        mm_pred_dic = get_movie_movie_pred(qmM, trainM, k, sim_arg, weight_arg, model_arg)
        mm_output(mm_pred_dic, mu_list, dev_filepath, output_filepath)

    # ==========/ EXP 3(pcc) /========== #
    if model_arg == 'pcc':
        mm_pred_dic = get_movie_movie_pred(qmM, trainM, k, sim_arg, weight_arg, model_arg)
        mm_output(mm_pred_dic, mu_list, dev_filepath, output_filepath)
    print time.time() - t0, "seconds wall time"


if __name__ == '__main__':
    main(sys.argv[1:])