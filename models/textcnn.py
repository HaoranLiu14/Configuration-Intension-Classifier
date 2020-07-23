import argparse
import time

import pandas as pd
from data.data_process import label2vec, vec2label, get_data
from keras.layers import Input, SpatialDropout1D, Conv1D, Flatten, Dense, MaxPooling1D, Dropout, concatenate
from keras.models import Model
from sklearn.metrics import accuracy_score


def train_textcnn(feature, label,
                  max_len=28, filters=256, filter_size=None, label_size=6):
    if filter_size is None:
        filter_size = [3, 4, 5]
    input_layer = Input(shape=(feature.shape[1], feature.shape[2]))
    text_embed = SpatialDropout1D(0.2)(input_layer)
    pool_layers = []

    for filter in filter_size:
        cnn_layer = Conv1D(filters, filter, padding='same', strides=1, activation='relu')(text_embed)
        pool_layer = MaxPooling1D(pool_size=max_len - filter + 1)(cnn_layer)
        pool_layers.append(pool_layer)

    cnn = concatenate(pool_layers, axis=-1)
    flat = Flatten()(cnn)
    drop = Dropout(0.2)(flat)
    output = Dense(label_size, activation='softmax')(drop)

    model = Model(input_layer, output)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    model.fit(feature, label,
              batch_size=32, epochs=50, verbose=2)
    return model


def test_model_by_probs(name, description, label, model):
    pred_name = model.predict(name)
    pred_desc = model.predict(description)
    pred = pred_name + pred_desc
    pred_label = vec2label(pred)
    acc = accuracy_score(label, pred_label)
    print(acc)
    return acc


def test_textcnn(feature, label, model):
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
    parser.add_argument("-m", "--merge_text", help="whether the name and description should be merged",
                        type=bool, default=False)
    parser.add_argument("-l", "--max_len", help="the max len of tokens", type=int, default=28)
    args = parser.parse_args()

    data_path = args.data_path
    vec_path = args.vec_path
    acc_path = args.res_path
    header = args.header
    random_state = args.random_state
    merge_text = args.merge_text
    max_len = args.max_len

    data = pd.read_csv(data_path)
    label = list(data['label'])
    label_size = len(data['label'].unique())

    print('Training with %s at' % header, time.asctime(time.localtime(time.time())))
    train_feature, train_label, test_feature, test_label = get_data(head=header, random_state=random_state,
                                                                    label=label, path=vec_path,
                                                                    merge_text=merge_text)
    train_label = label2vec(train_label, label_size)
    test_label = label2vec(test_label, label_size)
    model = train_textcnn(feature=train_feature, label=train_label,
                          label_size=label_size, max_len=max_len)
    acc = test_textcnn(test_feature, test_label, model)
    print('accuracy of %s is %.4f' % (header, acc))
    with open(acc_path, 'a') as f:
        f.write('%.4f,' % acc)
    print('Test end at', time.asctime(time.localtime(time.time())))
    print()
