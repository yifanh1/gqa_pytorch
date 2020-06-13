# From: https://github.com/ronilp/mac-network-pytorch-gqa/blob/bilinear_fusion/preprocess.py
# Edited: Add preprocessing: from 'train_all_questions_0.json' to 'train_all_questions_9.json'
import os
import sys
import json
import pickle

import nltk
# nltk.download('punkt')
import tqdm
from torchvision import transforms
from PIL import Image
# from transforms import Scale


def process_question(root, split, word_dic=None, answer_dic=None):
    if word_dic is None:
        word_dic = {}

    if answer_dic is None:
        answer_dic = {}
    # TODO: For our dataset, we need to add a loop for 'train_all_questions_0~9.json'
    result = []
    word_index = 1
    answer_index = 0
    for number in range(10):
        temp_result = []
        with open(os.path.join(root, 'questions', f'train_all_questions_{str(number)}.json')) as f:
            data = json.load(f)

        for question in tqdm.tqdm(data.values()):
            words = nltk.word_tokenize(question['question'])
            question_token = []

            for word in words:
                try:
                    question_token.append(word_dic[word])

                except:
                    question_token.append(word_index)
                    word_dic[word] = word_index
                    word_index += 1

            answer_word = question['answer']

            try:
                answer = answer_dic[answer_word]
            except:
                answer = answer_index
                answer_dic[answer_word] = answer_index
                answer_index += 1

            result.append((question['imageId'], question_token, answer))
            temp_result.append((question['imageId'], question_token, answer))
        with open(f'data/gqa_all_train_{str(number)}.pkl', 'wb') as f:
            pickle.dump(result, f)
        # ---imageID of the question-----list of tokens of question--answer index of answer dict

    with open(f'data/gqa_all_train.pkl', 'wb') as f:
        pickle.dump(result, f)
    # The python variable is stored in .pkl file
    return word_dic, answer_dic


if __name__ == '__main__':
    root = 'data'  # sys.argv[1]
    print('Start: Preprocess training data...')
    word_dic, answer_dic = process_question(root, 'train')
    # print('Start: Preprocess validating data...')
    # process_question(root, 'val', word_dic, answer_dic)

    with open('data/gqa_dic_all.pkl', 'wb') as f:
        pickle.dump({'word_dic': word_dic, 'answer_dic': answer_dic}, f)
