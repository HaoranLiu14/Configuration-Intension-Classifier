import argparse
import time
import re

import numpy as np
import pandas as pd
from data.data_process import label2vec, vec2label, get_data
from keras.layers import Input, SpatialDropout1D, Conv1D, Flatten, Dense, MaxPooling1D, Dropout, concatenate
from keras.models import Model
from sklearn.metrics import accuracy_score

def train_cic(train_name, train_feature, train_label,
              max_len_name=28, max_len_feature=28,
              filters_name=256, filters_feature=256,
              filter_size_name=None, filter_size_feature=None,
              label_size=6):
    if filter_size_name is None:
        filter_size_name = [3, 4, 5]
    if filter_size_feature is None:
        filter_size_feature = [3, 4, 5]

    input_layer_name = Input(shape=(train_name.shape[1], train_name.shape[2]))
    text_embed_name = SpatialDropout1D(0.2)(input_layer_name)
    pool_layers = []

    for filter_size in filter_size_name:
        cnn_layer = Conv1D(filters_name, filter_size, padding='same', strides=1, activation='relu')(text_embed_name)
        pool_layer = MaxPooling1D(pool_size=max_len_name - filter_size + 1)(cnn_layer)
        pool_layers.append(pool_layer)

    input_layer_feature = Input(shape=(train_feature.shape[1], train_feature.shape[2]))
    text_embed_feature = SpatialDropout1D(0.2)(input_layer_feature)

    for filter_size in filter_size_feature:
        cnn_layer = Conv1D(filters_feature, filter_size, padding='same', strides=1, activation='relu')(
            text_embed_feature)
        pool_layer = MaxPooling1D(pool_size=max_len_feature - filter_size + 1)(cnn_layer)
        pool_layers.append(pool_layer)

    cnn_layer = concatenate(pool_layers, axis=-1)
    flat_layer = Flatten()(cnn_layer)
    drop_layer = Dropout(0.2)(flat_layer)

    output = Dense(label_size, activation='softmax')(drop_layer)

    model = Model([input_layer_name, input_layer_feature], output)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    model.fit([train_name, train_feature], train_label,
              batch_size=32, epochs=50, verbose=2)
    return model


def test_cic(name, feature, label, model):
    pred = model.predict([name, feature])
    pred_label = vec2label(pred)
    acc = accuracy_score(label, pred_label)
    print(acc)
    return acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.description = 'Please input the path of annotated data, vectorized data and experiment results'
    parser.add_argument("-d", "--data_path", help="this is the path of annotated data")
    parser.add_argument("-v", "--vec_path", help="this is the path of vectorized data")
    parser.add_argument("-r", "--res_path", help="this is the path of experiment results")
    parser.add_argument("-hdr", "--header", help="this is the header of preprocessed data")
    parser.add_argument("-s", "--random_state", help="the random seed to split the dataset",
                        type=int, default=22)
    parser.add_argument("-m", "--merge_text", help="whether the name and description should be merged",
                        type=bool, default=False)
    parser.add_argument("-l", "--max_len", help="the max len of tokens",
                        type=int, default=28)
    parser.add_argument("-fsn", "--filter_size_name", help="the filter size list of name",
                        type=str, default='[1]')
    parser.add_argument("-fsd", "--filter_size_feature", help="the filter size list of feature",
                        type=str, default='[3 4 5]')
    args = parser.parse_args()

    data_path = args.data_path
    vec_path = args.vec_path
    acc_path = args.res_path
    header = args.header
    random_state = args.random_state
    merge_text = args.merge_text
    max_len = args.max_len
    filter_size_name = np.array([int(i) for i in re.findall(r'\d+', args.filter_size_name)])
    filter_size_feature = np.array([int(i) for i in re.findall(r'\d+', args.filter_size_feature)])

    print(filter_size_name)
    print(filter_size_feature )
    data = pd.read_csv(data_path)
    label = list(data['label'])
    label_size = len(data['label'].unique())
    print('Training with %s at' % header, time.asctime(time.localtime(time.time())))
    train_name, train_feature, train_label, \
    test_name, test_feature, test_label = get_data(head=header, random_state=random_state, label=label,
                                                   path=vec_path, split=True)
    train_label = label2vec(train_label, label_size)
    test_label = label2vec(test_label, label_size)
    model = train_cic(train_name, train_feature, train_label,
                      max_len_name=max_len, max_len_feature=max_len,
                      filters_name=256, filters_feature=256,
                      filter_size_name=filter_size_name, filter_size_feature=filter_size_feature,
                      label_size=label_size)
    acc = test_cic(test_name, test_feature, test_label, model)
    print('accuracy of %s is %.4f' % (header, acc))
    with open(acc_path, 'a') as f:
        f.write('%.4f,' % acc)
    print('Test end at', time.asctime(time.localtime(time.time())))
    print()
