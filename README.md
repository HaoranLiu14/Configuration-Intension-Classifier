# CIC
This is the source code of Configuration Intention Classifier (CIC) from paper "Deep Understanding of Configuration Intention".

## Text -> Tector
This part of work is supported by [bert-as-service](https://github.com/hanxiao/bert-as-service).
1. server side
There are 2 choices of converting texts to vectors. the parameter `max_len` should be a concrete number like 20, 30, etc.
- sentence to vector
```bash
bert-serving-start -model_dir uncased_L-12_H-768_A-12/ -num_worker=1 -max_seq_len=max_len
```
We use this command to change a sentence into a 768 vector.
- tokens to vector
```bash
bert-serving-start -model_dir uncased_L-12_H-768_A-12/ -num_worker=1 -pooling_strategy NONE -max_seq_len=max_len
```
We use this command to change a sentence into a max_len*768 vector.

In both cases, the parameter max_len points the number of words in a sentence that we should consider.

2. client side
Follow the instructions in [bert2vec.py](./data/bert2vec.py) to convert data in text form into vector form.
e.g. 
```bash
python data2vec.py -d ./dataset/data.csv -v ./vecdata/data -hdrs description
```

## model traing
In the model training stage, just follow the instructions in each source code file.
e.g. train CIC model, give the parameters needed for [cic.py](./models/cic.py), and run in command line like:
```bash
python cic.py -d ./dataset/data.csv -v ./vecdata/data -hdr description -r results.csv -s 22 -m False -l 30 -fsn [1] -fsd [3,4,5]
```