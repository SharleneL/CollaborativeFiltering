__author__ = 'luoshalin'

import numpy as np
from scipy import *


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


def uu_output(uu_pred_dic, mu_list, input_filepath, output_filepath):
    with open(output_filepath, 'a') as f_output:
        for mu_tuple in mu_list:
            mid = mu_tuple[0]
            uid = mu_tuple[1]
            res = uu_pred_dic[uid][mid]
            f_output.write(str(res)[0] + '\n')