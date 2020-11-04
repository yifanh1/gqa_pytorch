# Author: yifanh1
# Data: 6/10/2020
# Time: 2:35 PM
# return a numpy matrix
import pickle
import numpy as np

weights_matrix = None
vocab_size = 2950
answer_size = 1845


def get_embedding_weights():
    global weights_matrix, vocab_size, answer_size
    embedding_dim = 300
    if weights_matrix is not None:
        return weights_matrix
    with open('data/gqa_dic_all.pkl', 'rb') as f:
        dic = pickle.load(f)

    words = set(dic['word_dic'].keys())
    answers = set(dic['answer_dic'].keys())
    answer_size = len(answers)
    words.update(answers)
    vocab_size = len(words)
    vocab = {word: i for i, word in enumerate(words)}
    weights_matrix = np.zeros((vocab_size,embedding_dim))
    glove = {}
    with open('data/glove/glove.6B.300d.txt', 'r', encoding='UTF-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            glove[word] = coefs
    words_found = 0
    for i, word in enumerate(vocab):
        try:
            weights_matrix[i] = glove[word]
            words_found += 1
        except KeyError:
            weights_matrix[i] = np.random.normal(scale=0.6, size=(embedding_dim,))
    return weights_matrix


if __name__ == '__main__':
    w = get_embedding_weights()
    print(w[:10][:10])


