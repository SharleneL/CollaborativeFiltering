__author__ = 'luoshalin'

import numpy as np
from scipy.sparse import *
from scipy import *


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


def mm_output(mm_pred_dic, mu_list, dev_filepath, output_filepath):
    with open(output_filepath, 'a') as f_output:
        for mu_tuple in mu_list:
            mid = mu_tuple[0]
            uid = mu_tuple[1]
            res = mm_pred_dic[mid][uid]
            f_output.write(str(res)[0] + '\n')