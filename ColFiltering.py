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


def get_trainM(filepath):
    user = []   # as row
    movie = []  # as col
    score = []  # as data

    # saves the total number of users & movies
    user_size = 0
    movie_size = 0

    with open(filepath) as f:
        line = f.readline().strip()
        while line != '':
            movie_num = int(line.split(',')[0])
            user_num = int(line.split(',')[1])
            score_num = int(line.split(',')[2])

            movie.append(movie_num)
            user.append(user_num)
            score.append(score_num - 3)  # imputation option 2

            user_size = max(user_size, user_num)
            movie_size = max(movie_size, movie_num)

            line = f.readline().strip()
    trainM = csr_matrix((score, (user, movie)), shape=(user_size+1, movie_size+1))  # including 0
    return trainM


def get_qmM(filepath, mu_list):
    user = []   # as row
    movie = []  # as col
    score = []  # as data
    # saves the total number of users & movies
    user_size = 0
    movie_size = 0

    with open(filepath) as f:
        line = f.readline().strip()
        while line != '':
            movie_num = int(line.split(',')[0])
            user_num = int(line.split(',')[1])

            movie.append(movie_num)
            user.append(user_num)
            score.append(1)  # if value = 1, means this is a pos to be predicted
            mu_list.append((movie_num, user_num))

            user_size = max(user_size, user_num)
            movie_size = max(movie_size, movie_num)

            line = f.readline().strip()
    qmM = csr_matrix((score, (user, movie)), shape=(user_size+1, movie_size+1))  # including 0
    return qmM


def get_user_user_pred(qid_list, trainM, k, sim_arg, weight_arg):
    # calculate similarities - each row is the similarities for one target query, with all the other q's similarities(normalized)
    simM = get_uu_simM(qid_list, trainM, sim_arg)

    # set the sim of the same query to -1
    for i in range(len(qid_list)):
        qid = qid_list[i]
        simM[i, qid] = -1

    # get prediction result
    query_predict_res_dic = dict()
    # get the k-largest sims for query

    for i in range(0, len(simM)):
        # get the id list for k nearest queries
        sim_arr = np.array(simM[i])  # target query array
        temp = np.argpartition(-sim_arr, k)
        sim_id_list = temp[:k]
        knn_qid_list = sim_id_list[0][:k]  # k-nearest qids for current query(row in simM)

        # construct KNN matrix
        knn_M = trainM.tocsr()[knn_qid_list,:]
        sim_list = [simM[i, qid] for qid in knn_qid_list]

        if weight_arg == 'mean':  # mean
            predict_query_res = knn_M.mean(0) + 3  # a full list with converted imputation
            predict_query_res_list = list(np.array(predict_query_res[0, :]).reshape(-1,))
            predict_query_res_list = [round(x) for x in predict_query_res_list]

        if weight_arg == 'weight':  # weighted sum
            knn_arr_M = knn_M.toarray()
            res_list = [0] * len(knn_arr_M[0])
            # normalize
            sim_sum = sum(sim_list)
            w_list = sim_list
            if sim_sum != 0:
                w_list /= sim_sum
            w_arr = np.asarray(w_list)
            predict_query_res_list = dot(w_arr, knn_arr_M) + 3
            predict_query_res_list = [round(x) for x in predict_query_res_list]
        query_predict_res_dic[qid_list[i]] = predict_query_res_list
    print 'EXP1 END'
    return query_predict_res_dic


def get_uu_simM(qid_list, trainM, sim_arg):
    if sim_arg == 'dot':
        trainM_norm_arr = np.asarray([1] * trainM.shape[0])
        trainM = (trainM.T / trainM_norm_arr).T
        queryM = trainM[qid_list, :]
        return queryM * trainM.T

    if sim_arg == 'cosine':
        # normalize trainM
        trainM_norm_list = np.linalg.norm(trainM.toarray(), axis=1)  # the normalization factor for each row
        for i in range(len(trainM_norm_list)):
            if trainM_norm_list[i] == 0:
                trainM_norm_list[i] = 1
        trainM_norm = (trainM.T / trainM_norm_list).T  # the normalized trainM

        # get queryM - corresponding rows of qid_list
        queryM_norm = trainM_norm[qid_list, :]
        return dot(queryM_norm, trainM_norm.T)

# qmM - matrix to be predicted; trainM - original predicted matrix
def get_movie_movie_pred(qmM, trainM, k, sim_arg, weight_arg, model_arg):
    simM = get_mm_simM(trainM, sim_arg, model_arg)     # the movie-movie similarity matrix
    user_list = find(qmM)[0].tolist()       # the users to be predicted
    movie_list = find(qmM)[1].tolist()      # the movies to be predicted
    movie_set = set(movie_list)  # a set of mid
    score_list = []

    # knn
    movie_predict_res_dic = dict()  # saves <mid, list of queries' predictions>
    for mid in movie_set:
        # get the id list for k nearest movies
        sim_arr = np.array(simM[mid])  # target movie array
        temp = np.argpartition(-sim_arr, k)
        sim_id_list = temp[:k]
        knn_mid_list = sim_id_list[0][:k]  # k-nearest mids for current movie(row in simM)

        # construct KNN matrix
        knn_M = trainM.tocsr().T[knn_mid_list,:]
        sim_list = [simM[mid, knn_mid] for knn_mid in knn_mid_list]

        if weight_arg == 'mean':  # mean
            predict_movie_res = knn_M.mean(0) + 3  # a full list with reverse imputation
            predict_movie_res_list = list(np.array(predict_movie_res[0, :]).reshape(-1,))
            predict_movie_res_list = [round(x) for x in predict_movie_res_list]

        if weight_arg == 'weight':  # weighted sum
            knn_arr_M = knn_M.toarray()
            res_list = [0] * len(knn_arr_M[0])
            # normalize
            sim_sum = sum(sim_list)
            w_list = sim_list
            if sim_sum != 0:
                w_list /= sim_sum

            w_arr = np.asarray(w_list)
            predict_movie_res_list = dot(w_arr, knn_arr_M) + 3
            predict_movie_res_list = [round(x) for x in predict_movie_res_list]
        movie_predict_res_dic[mid] = predict_movie_res_list
    print 'EXP2 END'
    return movie_predict_res_dic


def get_mm_simM(trainM, sim_arg, model_arg):
    if model_arg == 'pcc':
        std_list = trainM.toarray().std(0)  # the std for each col(movie)
        std_M = np.matrix([std_list])
        return dot(std_M.T, std_M)
    else:
        if sim_arg == 'dot':
            trainM_norm_arr = np.asarray([1] * trainM.shape[0])
            trainM_norm = (trainM.T / trainM_norm_arr).T
            return dot(trainM_norm.T, trainM_norm)

        if sim_arg == 'cosine':
            # normalize trainM
            trainM_norm_list = np.linalg.norm(trainM.toarray(), axis=1)  # the normalization factor for each row
            for i in range(len(trainM_norm_list)):
                if trainM_norm_list[i] == 0:
                    trainM_norm_list[i] = 1
            trainM_norm = (trainM.T / trainM_norm_list).T  # the normalized trainM
            return dot(trainM_norm.T, trainM_norm)


def uu_output(uu_pred_dic, mu_list, input_filepath, output_filepath):
    with open(output_filepath, 'a') as f_output:
        for mu_tuple in mu_list:
            mid = mu_tuple[0]
            uid = mu_tuple[1]
            res = uu_pred_dic[uid][mid]
            f_output.write(str(res)[0] + '\n')


def mm_output(mm_pred_dic, mu_list, dev_filepath, output_filepath):
    with open(output_filepath, 'a') as f_output:
        for mu_tuple in mu_list:
            mid = mu_tuple[0]
            uid = mu_tuple[1]
            res = mm_pred_dic[mid][uid]
            f_output.write(str(res)[0] + '\n')


if __name__ == '__main__':
    main(sys.argv[1:])