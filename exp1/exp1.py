from typing import List
from nltk.tokenize import word_tokenize
from nltk import bigrams, FreqDist
from math import log
import re

train_path = './exp1/train_LM.txt'
test_path = './exp1/test_LM.txt'


def readfile(file_path: str) -> List:
    dialogues = []
    with open(file_path, 'r', encoding='utf-8') as f:
        dialogues = [re.sub(r'[^a-zA-Z0-9 ]', r'', text) for text in f.read().lower().split(sep='__eou__')]
    return dialogues


def perplexity(dataset: List, freq_dict: FreqDist, opt: str = None) -> float:
    ppt = []
    for sentence in dataset:
        log_prob, wt = 0, 0
        for word in word_tokenize(sentence) if opt == 'unigram' else bigrams(word_tokenize(sentence)):
            if word in freq_dict:
                log_prob += log(freq_dict[word], 2)
                wt += 1
        if wt > 0:
            ppt.append([sentence, pow(2, -(log_prob / wt))])
    ret = 0
    for ppt_tuple in ppt:
        ret += ppt_tuple[-1]
    return ret / len(ppt)


def uni_gram() -> None:
    unigrams_dist = FreqDist()
    unigrams_freq = FreqDist()
    #########
    # train #
    #########
    train = readfile(train_path)
    print(f"train_LM.txt loaded")
    for sentence in train:
        sen_word_freq = FreqDist(word_tokenize(sentence))  # process dataset sentence by sentence
        for word in sen_word_freq:
            if word in unigrams_dist:
                unigrams_dist[word] += sen_word_freq[word]
            else:
                unigrams_dist[word] = sen_word_freq[word]
    print(f"train dataset complete processing")
    ########
    # test #
    ########
    test = readfile(test_path)
    print(f"test_LM.txt loaded")
    for sentence in test:
        words = word_tokenize(sentence)
        for word in words:
            if word not in unigrams_dist:
                unigrams_dist[word] = 0
    print(f"test dataset complete processing")
    ##############
    # perplexity #
    ##############
    s = unigrams_dist.N() + unigrams_dist.B()
    # N()->the total number of sample outcomes recorded
    # B()->the total number of sample words that have counts greater than zero, B() is the same as len(FreqDist)
    for word in unigrams_dist:
        unigrams_freq[word] = (unigrams_dist[word] + 1) / s
    print(f"uni-gram perplexity is {perplexity(test, unigrams_freq, 'unigram')}")


def bi_gram():
    bigrams_dist = FreqDist()
    bigrams_freq = FreqDist()
    bigram, history = {}, {}
    #########
    # train #
    #########
    train = readfile(train_path)
    print(f"train_LM.txt loaded")
    for sentence in train:
        sen_word_freq = FreqDist(bigrams(word_tokenize(sentence)))
        for word in sen_word_freq:
            if word in bigrams_dist:
                bigrams_dist[word] += sen_word_freq[word]
            else:
                bigrams_dist[word] = sen_word_freq[word]
                if word[0] in bigram:
                    bigram[word[0]] += 1
                else:
                    bigram[word[0]] = 1
    print(f"train dataset complete processing")
    ########
    # test #
    ########
    test = readfile(test_path)
    print(f"test_LM.txt loaded")
    for sentence in test:
        words = bigrams(word_tokenize(sentence))
        for word in words:
            if word not in bigrams_dist:
                bigrams_dist[word] = 0
                if word[0] in bigram:
                    bigram[word[0]] += 1
                else:
                    bigram[word[0]] = 1
    print(f"test dataset complete processing")
    ##############
    # perplexity #
    ##############
    for word in bigrams_dist:
        if word[0] in history:
            history[word[0]] += bigrams_dist[word]
        else:
            history[word[0]] = bigrams_dist[word]
    for word in bigrams_dist:
        bigrams_freq[word] = (bigrams_dist[word] + 1) / (history[word[0]] + bigram[word[0]])
    print(f"uni-gram perplexity is {perplexity(test, bigrams_freq, 'bigram')}")


def main():
    uni_gram()
    bi_gram()


if __name__ == '__main__':
    main()
