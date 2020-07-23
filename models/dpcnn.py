import time

import pandas as pd
from keras.layers import Input, SpatialDropout1D, Conv1D, Flatten, Dense, Activation, Add, Concatenate
from keras.layers import PReLU, BatchNormalization
from keras.layers import GlobalMaxPool1D, MaxPooling1D
from keras.models import Model
from sklearn.metrics import accuracy_score
import argparse

from data.data_process import label2vec, vec2label, get_data


def Res_cnn(x, filters=256):
    x = Conv1D(filters=filters, kernel_size=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv1D(filters=filters, kernel_size=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    return x


def train_dpcnn(feature, label, filters=256, label_size=6):
    input_layer = Input(shape=(feature.shape[1], feature.shape[2]))
    text_embed = SpatialDropout1D(0.2)(input_layer)

    conv_1 = Conv1D(filters, kernel_size=1, padding='same')(text_embed)
    conv_1_prelu = PReLU()(conv_1)
    blk = Res_cnn(text_embed, filters=filters)
    block_add = Add()([blk, conv_1_prelu])
    blk = MaxPooling1D(pool_size=3, strides=2)(block_add)
    layer_cur = 0
    repeat = 3
    for i in range(repeat):
        if i == repeat - 1 or layer_cur == 1:
            block_last = Res_cnn(blk, filters=filters)
            block_add = Add()([block_last, blk])
            blk = GlobalMaxPool1D()(block_add)
        else:
            block_middle = Res_cnn(blk, filters=filters)
            block_add = Add()([block_middle, blk])
            blk = MaxPooling1D(pool_size=3, strides=2)(block_add)

    dense_layer = Dense(256, activation='relu')(blk)
    dense_layer = BatchNormalization()(dense_layer)
    output = Dense(label_size, activation='sigmoid')(dense_layer)

    model = Model(input_layer, output)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

    model.fit(feature, label, batch_size=32, epochs=50, verbose=2)

    return model

def test_dpcnn(feature, label, model):
    pred = model.predict(feature)
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
    parser.add_argument("-s", "--random_state", help="the random seed to split the dataset", default=22)
    args = parser.parse_args()

    data_path = args.data_path
    vec_path = args.vec_path
    acc_path = args.res_path
    header = args.header
    random_state = args.random_state

    data = pd.read_csv(data_path)
    label = list(data['label'])
    label_size = len(data['label'].unique())
    train_feature, train_label, test_feature, test_label = get_data(head=header, random_state=random_state,
                                                                    label=label, path=vec_path)
    train_label = label2vec(train_label, label_size)
    test_label = label2vec(test_label, label_size)
    model = train_dpcnn(train_feature, train_label, label_size=label_size)
    acc = test_dpcnn(test_feature, test_label, model)
    with open(acc_path, 'a') as f:
        f.write('%.4f,' % acc)
