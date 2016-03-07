# RUNNING COMMAND:
# python ColFiltering.py [exp1/exp2/exp3/exp4]

__author__ = 'luoshalin'

from scipy import sparse
import os
import sys
import time
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine
from scipy import spatial
import heapq
from numpy import linalg as LA

from scipy.sparse import *
from scipy import *


def main(argv):
    # PARAMETERS
    arg = sys.argv[0]
    dev_filepath = '../../data/HW4_data/dev.csv'
    test_filepath = '../../data/HW4_data/test.csv'
    train_filepath = '../../data/HW4_data/train.csv'
    dev_query_filepath = '../../data/HW4_data/dev.queries'
    test_query_filepath = '../../data/HW4_data/test.queries'
    output_filepath = 'output'
    k = 10  # default

    # TEST -- START
    train_filepath = '../../data/HW4_data/train.csv'
    dev_query_filepath = '../../data/HW4_data/dev_s.queries'
    # TEST -- END

    # READ DATA
    # read train data -> preprocessing into vectors -> imputation
    trainM = get_trainM(train_filepath)  # get a sparse M as training set, <user, movie>[score]
    # print trainM[19].toarray()[0][1]
    # CORRECT

    # read dev query ids
    qid_set = get_query_set(dev_filepath)  # get a list of target query ids
    qid_list = list(qid_set)
    # run user-user similarity algo
    uu_dic = get_user_user_sim(qid_list, trainM, k, 'm')
    # write to result
    # output(uuM, dev_filepath, output_filepath)


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


def get_query_set(filepath):  # impunity??
    qid_set = set()
    with open(filepath) as f:
        line = f.readline().strip()
        while line != '':
            qid_set.add(int(line.split(',')[0]))
            line = f.readline().strip()
    return qid_set

# for exp1
def get_user_user_sim(qid_list, trainM, k, method):
    # normalize trainM
    trainM_norm_list = np.linalg.norm(trainM.toarray(), axis=1)  # the normalization factor for each row
    for i in range(len(trainM_norm_list)):
        if trainM_norm_list[i] == 0:
            trainM_norm_list[i] = 1
    trainM_norm = (trainM.T / trainM_norm_list).T  # the normalized trainM

    # get queryM - corresponding rows of qid_list
    queryM_norm = trainM_norm[qid_list, :]

    # calculate similarities - each row is the similarities for one target query, with all the other q's similarities(normalized)
    simM = dot(queryM_norm, trainM_norm.T)

    # set the sim of the same query to -inf
    for i in range(len(qid_list)):
        qid = qid_list[i]
        simM[i, qid] = -inf

    # get prediction result
    query_predict_res_dic = dict()
    # get the k-largest sims for query

    for i in range(0, len(simM)):
        sim_arr = np.array(simM[i])  # target query array
        temp = np.argpartition(-sim_arr, k)
        sim_id_list = temp[:k]
        # sim_id_list = [x+1 for x in sim_id_list]
        # print sim_id_list
        knn_qid_list = sim_id_list[0][:k]  # k-nearest qids for current query(row in simM)

        # construct KNN matrix
        # knn_M = trainM[knn_qid_list, :]
        knn_M = trainM.tocsr()[knn_qid_list,:]
        sim_list = [simM[i, qid] for qid in knn_qid_list]
        # knn_M = []  # a list of numpy arrays
        # sim_list = []  # a list of similarity of the corresponding numpy array element
        # for qid in knn_qid_list:
        #     # print id
        #     # print trainM.toarray(id)[0]
        #     knn_M.append(trainM.toarray(qid)[0])
        #     sim_list.append(simM[i, qid])

        if method == 'm':  # mean
            predict_query_res = knn_M.mean(0) + 3  # a full list with converted imputation
            predict_query_res_list = list(np.array(predict_query_res[0, :]).reshape(-1,))
            predict_query_res_list = [round(x) for x in predict_query_res_list]
            # print knn_M
            # print predict_query_res

        if method == 'w':  # weighted sum
            res_list = [0] * len(knn_M[0])
            # normalize
            w_list = sim_list / sum(sim_list)
            for i in range(0, len(w_list)):
                tmp_res_list = knn_M[i] * w_list[i]
                predict_query_res = [round(x+y+3) for x, y in zip(res_list, tmp_res_list)]
        # reverse imputation
        # predict_query_res_list = [round(x)+3 for x in predict_query_res]
        query_predict_res_dic[qid_list[i]] = predict_query_res_list
    print 'END'
    return query_predict_res_dic












    # query_predict_res_dic = dict()  # a dict saves <qid, q_predict_res>
    #
    #
    #
    # for qid in qid_set:  # process one query
    #     print qid
    #     query = trainM[qid].toarray()  # [0]
    #     # print query
    #     # find the k-nearest qids in trainM for current query  - by getting a size K heap with K most similar row-points
    #     heap = []
    #     # get the sim between query & all vectors in matrix
    #     sim_list = []
    #     for i in range(0, trainM.shape[0]):  # each row
    #         if i == qid:
    #             continue
    #         if LA.norm(trainM[i].toarray()) == 0:
    #             sim = -inf
    #         else:
    #             sim = 1 - spatial.distance.cosine(query, trainM[i].toarray())  # cosine sim - higher is more similar
    #         sim_list.append(sim)
    #     sim_arr = np.array(sim_list)
    #
    #     temp = np.argpartition(-sim_arr, k)
    #     sim_id_list = temp[:k]
    #     sim_id_list = [x+1 for x in sim_id_list]
    #     print sim_id_list


        # for i in range(0, trainM.shape[0]):  # each row
        #     if i == qid:
        #         continue
        #     if LA.norm(trainM[i].toarray()) == 0:
        #         sim = -inf
        #     else:
        #         sim = 1 - spatial.distance.cosine(query, trainM[i].toarray())  # cosine sim - higher is more similar
        #     heapq.heappush(heap, (sim, i))  # <similarity, rowid@M>
        #     if len(heap) > k:
        #         heapq.heappop(heap)
        # print heap
        # CORRECT

        # construct KNN matrix
    #     knn_M = []  # a list of numpy arrays
    #     sim_list = []  # a list of similarity of the corresponding numpy array element
    #     for sim_id in sim_id_list:
    #         # print id
    #         # print trainM.toarray(id)[0]
    #         knn_M.append(trainM.toarray(sim_id)[0])
    #         sim_list.append(sim)
    #
    #     if method == 'm':  # mean
    #         predict_query_res = np.mean(knn_M, axis=0)  # a full list without imputation
    #         # print knn_M
    #         # print predict_query_res
    #
    #     if method == 'w':  # weighted sum
    #         res_list = [0] * len(knn_M[0])
    #         # normalize
    #         w_list = sim_list / sum(sim_list)
    #         for i in range(0, len(w_list)):
    #             tmp_res_list = knn_M[i] * w_list[i]
    #             predict_query_res = [x+y for x, y in zip(res_list, tmp_res_list)]
    #     # reverse imputation
    #     predict_query_res = [round(x)+3 for x in predict_query_res]
    #     query_predict_res_dic[qid] = predict_query_res
    # return query_predict_res_dic


# input file : dev file
def output(uu_dic, input_filepath, output_filepath):
    with open(output_filepath, 'a') as f_output:
        with open(input_filepath) as f:
            line = f.readline().strip()
            while line != '':
                qid = int(line.split(',')[0])
                mid = int(line.split(',')[1])
                res = uu_dic[qid][mid]
                new_line = str(qid) + ',' + str(mid) + ',' + str(res) + '\n'
                f_output.write(new_line)


if __name__ == '__main__':
    main(sys.argv[1:])