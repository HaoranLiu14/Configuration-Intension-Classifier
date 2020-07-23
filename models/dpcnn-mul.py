import time

import pandas as pd
from keras.layers import Input, SpatialDropout1D, Conv1D, Flatten, Dense, Activation, Add, Concatenate
from keras.layers import PReLU, BatchNormalization
from keras.layers import GlobalMaxPool1D, MaxPooling1D
from keras.models import Model
from sklearn.metrics import accuracy_score
import argparse

from data.data_process import label2vec, vec2label, get_data


def block(pre, num_filters):
    x = Activation(activation='relu')(pre)
    x = Conv1D(filters=num_filters, kernel_size=3, padding='same', strides=1)(x)
    x = Activation(activation='relu')(x)
    x = Conv1D(filters=num_filters, kernel_size=3, padding='same', strides=1)(x)
    x = Add()([x, pre])
    return x


def mul_model(name, description, label, filters=256, label_size=6):
    input_layer_name = Input(shape=(name.shape[1], name.shape[2]))
    name_embed = SpatialDropout1D(0.2)(input_layer_name)
    region_x_name = Conv1D(filters=filters, kernel_size=3, padding='same', strides=1)(name_embed)
    x_name = block(region_x_name, num_filters=filters)
    for _ in range(3):
        px = MaxPooling1D(pool_size=3, strides=2, padding='same')(x_name)
        x_name = block(px, num_filters=filters)
    x_name = MaxPooling1D(pool_size=1)(x_name)
    sentence_embed_name = Flatten()(x_name)
    dense_layer_name = Dense(256, activation='relu')(sentence_embed_name)
    dense_name = Dense(label_size, activation='sigmoid')(dense_layer_name)

    input_layer_description = Input(shape=(description.shape[1], description.shape[2]))
    description_embed = SpatialDropout1D(0.2)(input_layer_description)
    region_x_description = Conv1D(filters=filters, kernel_size=3, padding='same', strides=1)(description_embed)
    x_description = block(region_x_description, num_filters=filters)
    for _ in range(3):
        px = MaxPooling1D(pool_size=3, strides=2, padding='same')(x_description)
        x_description = block(px, num_filters=filters)
    x_description = MaxPooling1D(pool_size=1)(x_description)
    sentence_embed_description = Flatten()(x_description)
    dense_layer_description = Dense(256, activation='relu')(sentence_embed_description)
    dense_description = Dense(label_size, activation='sigmoid')(dense_layer_description)

    output = Add()([dense_name, dense_description])

    model = Model([input_layer_name, input_layer_description], output)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

    model.fit([name, description], label, batch_size=32, epochs=50, verbose=1)

    return model


def test_mul_model(name, description, label, model):
    pred = model.predict([name, description])
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
    parser.add_argument("-h", "--header", help="this is the header of preprocessed data")
    parser.add_argument("-s", "--random_state", help="the random seed to split the dataset", default=22)
    args = parser.parse_args()

    data_path, vec_path, acc_path, header, random_state = args.data_path, args.vec_path, args.res_path, args.header, args.random_state

    data = pd.read_csv(data_path)
    label = list(data['label'])
    label_size = len(data['label'].unique())

    train_name, train_feature, train_label, \
    test_name, test_feature, test_label = get_data(head=header, random_state=random_state, label=label,
                                                   path=vec_path, split=True)
    train_name = train_name.reshape((train_name.shape[0], 1, train_name.shape[1]))
    test_name = test_name.reshape((test_name.shape[0], 1, test_name.shape[1]))
    train_feature = train_feature.reshape((train_feature.shape[0], 1, train_feature.shape[1]))
    test_feature = test_feature.reshape((test_feature.shape[0], 1, test_feature.shape[1]))
    train_label = label2vec(train_label, label_size)
    model = mul_model(train_name, train_feature, train_label, label_size=label_size)
    acc = test_mul_model(test_name, test_feature, test_label, model)
    print('accuracy of %s is %.4f' % (header, acc))
    with open(acc_path, 'a') as f:
        f.write('%.4f,' % acc)
    print('Test end at', time.asctime(time.localtime(time.time())))
    print()
