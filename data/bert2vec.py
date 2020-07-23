from bert_serving.client import BertClient
import os
import pandas as pd
import numpy as np
import argparse
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.description = 'Please input the path of annotated data and vectorized data'
    parser.add_argument("-d", "--data_path", help="this is the path of annotated data")
    parser.add_argument("-v", "--vec_path", help="this is the path of vectorized data")
    parser.add_argument("-hdrs", "--headers", help="this is the header of preprocessed data")
    args = parser.parse_args()

    data_path = args.data_path
    vec_path = args.vec_path
    headers = args.headers.split()

    data = pd.read_csv(data_path)

    bc = BertClient(check_length=False)

    for header in headers:
        print(header)
        header_vec = bc.encode(list(data[header]))
        header_path = os.path.join(vec_path, header+'.npy')
        print(header_path)
        np.save(header_path, header_vec)
