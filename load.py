"""
    load(by_word=False, by_sent=False, numeric_label=False, binary=False)

    example:

        X_train, y_train, X_test, y_test = load()

    X_train and X_test will be lists of tuples (headline, article)
    y_train and y_test will be lists of labels

    default types:
        headline: string
        article: string
        label: string

    options:
        by_word=True:       headline and article will be tokenized as a list of lower-case words.
        by_sent=True:       article will be a list of strings, 1 string for each sentence.
                            If combined with by_word=True, article will be a list of lists of strings.
        numeric_label=True: y_train, y_test will be numpy array where 0 = unrelated,
                            1 = discuss, 2 = agree, 3 = disagree.
        binary=True:        if numeric_label=True, y_train, y_test will be numpy array 0 = unrelated and 1 = otherwise.

"""
import pandas as pd
from nltk.tokenize.punkt import PunktSentenceTokenizer
from nltk.tokenize import TreebankWordTokenizer
import multiprocessing
import numpy as np
import json
import random


NUM_CPUS = multiprocessing.cpu_count()
pool = multiprocessing.Pool(processes=NUM_CPUS)
ITERS = (list, tuple, pd.core.series.Series)

TRAIN_BODIES = './data/train_bodies.csv'
TRAIN_STANCES = './data/train_stances.csv'
TEST_TRAIN_SPLIT = './data/test_train_ids.json'

s_tokenizer = PunktSentenceTokenizer()
w_tokenizer = TreebankWordTokenizer()


def parallelize(f, arg):
    out = pool.map(f, arg)
    return out


def word_tokenize(doc):
    if isinstance(doc, ITERS):
        if isinstance(doc[0], ITERS):
            return [word_tokenize(d) for d in doc]
        else:
            doc = [d.lower() for d in doc]
            return parallelize(w_tokenizer.tokenize, doc)
    else:
        return w_tokenizer.tokenize(doc.lower())


def sent_tokenize(doc):
    if isinstance(doc, ITERS):
        if isinstance(doc[0], ITERS):
            return [sent_tokenize(d) for d in doc]
        return pool.map(s_tokenizer.tokenize, doc)
    else:
        return s_tokenizer.tokenize(doc)


def load(by_word=False, by_sent=False, numeric_label=False, binary=False):
    train_bodies = pd.read_csv(TRAIN_BODIES)
    train_stances = pd.read_csv(TRAIN_STANCES)

    body_ids = train_bodies.loc[:, 'Body ID']
    articles = train_bodies['articleBody']
    headlines = train_stances['Headline']
    stances = train_stances.loc[:, 'Stance']

    if by_sent:
        articles = sent_tokenize(articles)
    if by_word:
        articles = word_tokenize(articles)
        headlines = word_tokenize(headlines)
    if numeric_label:
        if binary:
            label_map = {'agree': 1, 'disagree': 1, 'discuss': 1, 'unrelated': 0}
        else:
            label_map = {'unrelated': 0, 'discuss': 1, 'agree': 2, 'disagree': 3}
        stances = [label_map[s] for s in stances]

    bodyid_to_article = dict(zip(train_bodies['Body ID'], articles))
    id_to_headline = dict(enumerate(headlines))
    id_to_bodyid = dict(enumerate(train_stances['Body ID']))
    id_to_stance = dict(enumerate(stances))

    train_X = []
    train_y = []
    test_X = []
    test_y = []

    with open(TEST_TRAIN_SPLIT) as f:
        test_train_split = json.load(f)

    train_ids = set(test_train_split['train'])
    test_ids = set(test_train_split['test'])

    for i, headline in id_to_headline.items():
        if i in train_ids:
            train_X.append((id_to_headline[i], bodyid_to_article[id_to_bodyid[i]]))
            train_y.append(id_to_stance[i])
        else:
            test_X.append((id_to_headline[i], bodyid_to_article[id_to_bodyid[i]]))
            test_y.append(id_to_stance[i])

    if numeric_label:
        train_y = np.array(train_y)
        test_y = np.array(test_y)

    return train_X, train_y, test_X, test_y


def test():

    train_X, train_y, test_X, test_y = load(True, True, True, True)

    i = random.choice(range(len(test_y)))
    print('\nRandom train article (headline, article):\n\n{}\n\nLabel: {}\n'.format(train_X[i], train_y[i]))
    print('\nRandom test article (headline, article):\n\n{}\n\nLabel: {}\n'.format(test_X[i], test_y[i]))


if __name__ == '__main__':
    test()