import time

import pandas as pd
from keras.layers import Input, SpatialDropout1D, Conv1D, Flatten, Dense, Activation, Add, Concatenate
from keras.layers import PReLU, BatchNormalization
from keras.layers import GlobalMaxPool1D, MaxPooling1D
from keras.models import Model
from keras.layers import Dense, LSTM, Dropout, Permute, MaxPooling1D
from sklearn.metrics import accuracy_score
import argparse
import keras.backend as K
import keras.activations

from keras import Sequential

from data.data_process import label2vec, vec2label, get_data


def train_textrcnn(feature, label, label_size=6, sen_len=28):
    input_layer = Input(shape=(feature.shape[1], feature.shape[2]))
    text_embed = SpatialDropout1D(0.2)(input_layer)

    rnn_pro = LSTM(input_dim=768, output_dim=128)(text_embed)
    newinput = K.concatenate((text_embed, rnn_pro), 2)
    blk = keras.activations.relu(newinput, alpha=0.0, max_value=None, threshold=0.0)
    blk = Permute((2,1))(blk)
    compressed_v = MaxPooling1D(sen_len)(blk)
    output = Dense(label_size)(compressed_v)

    model = Model(input=[input_layer], output=output)

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

    model.fit(feature, label, batch_size=32, epochs=50, verbose=2)

    return model

def test_textrcnn(feature, label, model):
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
    model = train_textrcnn(train_feature, train_label, label_size=label_size)
    acc = test_textrcnn(test_feature, test_label, model)
    with open(acc_path, 'a') as f:
        f.write('%.4f,' % acc)
