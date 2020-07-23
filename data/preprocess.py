import pandas as pd
from nltk import PorterStemmer
import re
import enchant
import pickle


def check(word):
    dic = enchant.Dict("en_US")
    if word == '':
        return ''
    if dic.check(word):
        return word
    else:
        for w in dic.suggest(word):
            if word in w:
                return w
    return word


def stemmer(tokens):
    ps = PorterStemmer()
    tks = [ps.stem(word) for word in tokens]
    return [check(word) for word in tks]


def preprocess(description, except_word='', stop_words=None, stemming=False):
    print(description)
    if stop_words is None:
        stop_words = set(stopwords.words('english'))
    pattern = r',|\.|/|;|\'|`|\[|\]|<|>|\?|:|"|\{|\}|\~|!|@|#|\$|%|\^|&|\(|\)|-|=|\_|\+|，|。|、|；|‘|’|【|】|·|！| |…|（|）'
    tokens = re.split(pattern, description)
    # change to lower format
    tokens1 = [word.lower() for word in tokens]
    # remove stopwords
    tokens2 = []
    for word in tokens1:
        if word == except_word:
            tokens2.append(word)
        elif word not in stop_words:
            tokens2.append(word)

    if stemming:
        tokens2 = stemmer(tokens2)
    print(' '.join(tokens2))
    return ' '.join(tokens2)


def get_words(file):
    data = pd.read_csv(file)
    word_fre_dict = {}

    pattern = r'\w+(?:\.\w+)*(?:\(\w*\))?'
    for sent in list(data['description']):
        tokens = re.findall(pattern, sent.lower())
        for token in tokens:
            try:
                word_fre_dict[token] += 1
            except KeyError:
                word_fre_dict[token] = 1
    return word_fre_dict


# word_file 里面的单词已经按照字典序排好
def gen_word_dict(file):
    word_fre_dict = get_words(file)
    token_seq = list(word_fre_dict.keys())
    token_seq.sort()
    ps = PorterStemmer()
    dic = enchant.Dict("en_US")
    words = {}
    unknown = set()
    for i in range(len(token_seq)):
        word = token_seq[i]
        fre = word_fre_dict[word]

        # match words that contain numbers, mainly integer, float number, address
        pattern_int = r'^\d+$'
        pattern_float = r'^\d+\.\d+$'
        pattern_address = r'^\d{1,3}(\.\d{1,3}){3}$'
        if re.match(pattern_int, word) is not None:
            words[word] = 'number'
            continue
        if re.match(pattern_float, word) is not None:
            words[word] = 'float number'
            continue
        if re.match(pattern_address, word) is not None:
            words[word] = 'address'
            continue

        # handle other words
        # 2.1 whether this word is a combination of a few words
        word_list = re.findall(r'[a-zA-Z0-9]+', word)
        if len(word_list) == 0:
            print(word)
            words[word] = ''
            continue
        if len(word_list) > 1:
            words[word] = ' '.join(word_list)
            continue
        # 2.2 whether the former word is the origin of this word,
        # generally the former word should have more than 4 words
        pre_word = words[token_seq[i - 1]]
        if pre_word in word_list[0] and len(pre_word) > 4:
            words[word] = pre_word
            continue
        # 2.3 filter words with low frequency
        else:
            if word_list[0] not in vocab:
                unknown.add(word_list[0])
            words[word] = word_list[0]
    with open('words_newest.csv', 'w') as f:
        for key in token_seq:
            f.write('%s,%s,%d\n' % (key, words[key], word_fre_dict[key]))
    with open('unknown.csv', 'w') as f:
        unknown_words = sorted(list(unknown), key=lambda x:len(x))
        f.write('\n'.join(unknown_words))
    return words


def sub(sent, word_dict):
    print(sent)
    pattern = r'\w+(?:\.\w+)*(?:\(\w*\))?'
    tokens = re.findall(pattern, sent.lower())
    res = []
    for token in tokens:
        try:
            word = word_dict[token]
            res.append(word)
        except KeyError:
            word = ''
    print(' '.join(res))
    return ' '.join(res)
