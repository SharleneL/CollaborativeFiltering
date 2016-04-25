__author__ = 'luoshalin'


from scipy.sparse import *
from scipy import *


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