import pickle
from keras.preprocessing.text import Tokenizer
from data.Constants import label_dictionary, output_dictionary
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import re
import numpy as np


def train2vec(train, max_word=None, max_len=None):
    if max_len is None:
        max_len = [60, 10]
    if max_word is None:
        max_word = [1000, 1000]
    texts = train['description'].apply(preprocess)
    tokenizer = Tokenizer(num_words=max_word[0])
    tokenizer.fit_on_texts(texts=texts)

    # vectorize description
    desc = tokenizer.texts_to_sequences(train['description'].apply(preprocess))
    desc_vec = pad_sequences(maxlen=max_len[0], sequences=desc, padding='post', value=0)

    # vectorize configuration names
    name = tokenizer.texts_to_sequences(train['name'].apply(lambda x: re.split('[-._]', x.strip().lower())))
    name_vec = pad_sequences(maxlen=max_len[1], sequences=name, padding='post', value=0)

    with open("token.pk", 'wb') as token:
        pickle.dump([tokenizer, max_len], token)

    return desc_vec, name_vec


def test2vec(test, path="token.pk"):
    with open(path, 'rb') as token:
        data = pickle.load(token)
        tokenizer, max_len = data[0], data[1]
    # vectorize description
    desc = tokenizer.texts_to_sequences(test['description'].apply(preprocess))
    desc_vec = pad_sequences(maxlen=max_len[0], sequences=desc, padding='post', value=0)
    # vectorize configuration names
    name = tokenizer.texts_to_sequences(test['name'].apply(lambda x: re.split('[-._]', x.strip().lower())))
    name_vec = pad_sequences(maxlen=max_len[1], sequences=name, padding='post', value=0)

    return desc_vec, name_vec


# return one-hot encode for label
def one_hot(num, label_size):
    res = np.zeros(label_size)
    res[num] += 1
    return res


def label2vec(labels, label_size):
    label_vec = np.array([one_hot(label_dictionary[label], label_size) for label in labels])
    return label_vec


def vec2label(label_vec):
    res = [output_dictionary[np.argmax(p)] for p in label_vec]
    return res


def name2tokens(name):
    pattern = r'[A-Z]*[a-z]+'
    tokens = re.findall(pattern, name)
    return ' '.join([token.lower() for token in tokens])


def merge(data, header):
    description = list(data[header])
    name = list(data['name'])
    res = []
    for i in range(len(data)):
        n = name2tokens(name[i])
        res.append(n + ' ' + description[i])
    return res


# predict label by probability
def prob2label(proba, threshold):
    """
    proba: list of probabilities
    threshold: float number in [0,1]
    """
    res = []
    for p in proba:
        if max(p) < threshold:
            res.append('manageability')
        else:
            res.append(output_dictionary[np.argmax(p)])
    return res


def get_data(head, random_state, label, path, split=False, merge_text=False):
    if not merge_text:
        name = np.load(path + 'name' + '.npy')
        head_vec = np.load(path + head + '.npy')
        if not split:
            feature = np.concatenate((name, head_vec), axis=1)
            train_feature, test_feature, train_label, test_label = train_test_split(feature, label, test_size=0.1,
                                                                                    random_state=random_state)
            return train_feature, train_label, test_feature, test_label
        else:
            train_name, test_name, \
            train_description, test_description, \
            train_label, test_label = train_test_split(name, head_vec,
                                                       label, test_size=0.1, random_state=random_state)
            return train_name, train_description, train_label, test_name, test_description, test_label
    else:
        feature = np.load(path + head + '.npy')
        train_feature, test_feature, train_label, test_label = train_test_split(feature, label, test_size=0.1,
                                                                                random_state=random_state)
        return train_feature, train_label, test_feature, test_label


def get_data_k(data, random_state, max_word=1000, max_len_desc=28, max_len_name=28):
    """
    embedding by keras
    """
    train, test = train_test_split(data, test_size=0.1, random_state=random_state)

    # generate dictionary for description
    texts = train['description']
    tokenizer = Tokenizer(num_words=max_word)
    tokenizer.fit_on_texts(texts=texts)
    index_word = tokenizer.index_word  # dict:{number: word}
    word_index = tokenizer.word_index  # dict:{word: number}
    # vectorize description
    train_description = tokenizer.texts_to_sequences(train['description'].apply(preprocess))
    train_description = pad_sequences(maxlen=max_len_desc, sequences=train_description, padding='post', value=0)
    test_description = tokenizer.texts_to_sequences(test['description'].apply(preprocess))
    test_description = pad_sequences(maxlen=max_len_desc, sequences=test_description, padding='post', value=0)

    # vectorize configuration names
    train_name = tokenizer.texts_to_sequences(train['name'].apply(lambda x: re.split('[-._]', x.strip().lower())))
    train_name = pad_sequences(maxlen=max_len_name, sequences=train_name, padding='post', value=0)
    test_name = tokenizer.texts_to_sequences(test['name'].apply(lambda x: re.split('[-._]', x.strip().lower())))
    test_name = pad_sequences(maxlen=max_len_name, sequences=test_name, padding='post', value=0)

    train_label = list(train['label'])
    test_label = list(test['label'])

    return train_name, train_description, train_label, \
           test_name, test_description, test_label


def get_data_by_header(header, path, random_state, label):
    feature = np.load(path + header + '.npy')
    train_feature, test_feature, train_label, test_label = train_test_split(feature, label, test_size=0.1,
                                                                            random_state=random_state)
    return train_feature, train_label, test_feature, test_label