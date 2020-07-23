import time

import pandas as pd
from keras.layers import Input, SpatialDropout1D, Conv1D, Flatten, Dense, Activation, Add, Concatenate
from keras.layers import PReLU, BatchNormalization
from keras.layers import GlobalMaxPool1D, MaxPooling1D
from keras.models import Model
from keras.layers import Dense, LSTM, Dropout, Permute, Reshape, Lambda, RepeatVector, Multiply
from sklearn.metrics import accuracy_score
import argparse
import keras.backend as K

from keras import Sequential

from data.data_process import label2vec, vec2label, get_data

def att(inputs, sen_len, input_dim):
    # inputs.shape = (batch_size, sen_len, input_dim)
    a = Permute((2, 1))(inputs)
    a = Dense(sen_len, activation='softmax')(a)

    a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
    a = RepeatVector(input_dim)(a)

    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul


def train_textrnn_att(feature, label, label_size=6, sen_len=30, input_dim=768):
    inputs = Input(shape=(sen_len, input_dim,))
    lstm_units = 32
    lstm_out = LSTM(lstm_units, return_sequences=True)(inputs)
    attention_mul = att(lstm_out)
    attention_mul = Flatten()(attention_mul)
    output = Dense(label_size, activation='sigmoid')(attention_mul)
    model = Model(input=[inputs], output=output)

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    model.fit(feature, label, batch_size=32, epochs=50, verbose=2)

    return model

def test_textrnn_att(feature, label, model):
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
    model = train_textrnn_att(train_feature, train_label, label_size=label_size)
    acc = test_textrnn_att(test_feature, test_label, model)
    with open(acc_path, 'a') as f:
        f.write('%.4f,' % acc)
