import argparse
import time

import numpy as np
import pandas as pd
from data.Constants import output_dictionary, label_dictionary
from data.data_process import get_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def train_rfc(train_feature, train_label):
    train_label = [label_dictionary[tl] for tl in train_label]
    model = RandomForestClassifier(random_state=0)
    model.fit(train_feature, train_label)
    return model


def test_model(feature, label, model):
    prob = model.predict_proba(feature)
    pred_label = [output_dictionary[np.argmax(p)] for p in prob]
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

    print('Training with %s at' % header, time.asctime(time.localtime(time.time())))
    train_feature, train_label, test_feature, test_label = get_data(head=header, random_state=random_state,
                                                                    label=label, path=vec_path)
    model = train_rfc(train_feature, train_label)
    acc = test_model(test_feature, test_label, model)
    print('accuracy of %s is %.4f' % (header, acc))
    with open(acc_path, 'a') as f:
        f.write('%.4f,' % acc)
    print('Test end at', time.asctime(time.localtime(time.time())))
    print()