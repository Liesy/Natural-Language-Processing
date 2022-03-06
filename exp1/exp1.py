from typing import List
from nltk.tokenize import word_tokenize
from nltk import bigrams, FreqDist
from math import log
import re

train_path = './exp1/train_LM.txt'
test_path = './exp1/test_LM.txt'


def readfile(file_path: str) -> List:
    with open(file_path, 'r', encoding='utf-8') as f:
        dialogues = [re.sub(r'[^a-zA-Z0-9 ]', r'', text) for text in f.read().lower().split(sep='__eou__')]
    return dialogues


def perplexity(dataset: List, freq_dict: FreqDist, opt: str = None) -> float:
    assert opt is not None
    ppl_sentence = []
    for sentence in dataset:
        log_prob, word_count = 0, 0
        for word in word_tokenize(sentence) if opt == 'unigram' else bigrams(word_tokenize(sentence)):
            if word in freq_dict:
                log_prob += log(freq_dict[word], 2)
                word_count += 1
        if word_count > 0:
            ppl_sentence.append([sentence, pow(2, -(log_prob / word_count))])
    sum_ppl = 0
    for ppt_tuple in ppl_sentence:
        sum_ppl += ppt_tuple[-1]
    return sum_ppl / len(ppl_sentence)


def uni_gram() -> None:
    unigrams_dist = FreqDist()
    unigrams_freq = FreqDist()
    #########
    # train #
    #########
    train = readfile(train_path)
    print(f"train_LM.txt loaded")
    for sentence in train:
        sen_word_freq = FreqDist(word_tokenize(sentence))
        # process dataset sentence by sentence
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
    # B()->the total number of sample words that have counts greater than zero
    # B() is the same as len(FreqDist)
    for word in unigrams_dist:
        unigrams_freq[word] = (unigrams_dist[word] + 1) / s
    print(f"uni-gram perplexity is {perplexity(test, unigrams_freq, 'unigram')}")


def bi_gram() -> None:
    bigrams_dist = FreqDist()
    bigrams_freq = FreqDist()
    word_begin, word_end = {}, {}  # 分别存放以word开头或word结尾的2-gram的种类数量
    #########
    # train #
    #########
    train = readfile(train_path)
    print(f"train_LM.txt loaded")
    for sentence in train:
        sen_bigram_freq = FreqDist(bigrams(word_tokenize(sentence)))
        for bigram in sen_bigram_freq:
            if bigram in bigrams_dist:
                bigrams_dist[bigram] += sen_bigram_freq[bigram]
            else:
                bigrams_dist[bigram] = sen_bigram_freq[bigram]
                if bigram[0] in word_begin:
                    word_begin[bigram[0]] += 1
                else:
                    word_begin[bigram[0]] = 1
    print(f"train dataset complete processing")
    ########
    # test #
    ########
    test = readfile(test_path)
    print(f"test_LM.txt loaded")
    for sentence in test:
        sen_bigrams = bigrams(word_tokenize(sentence))
        for bigram in sen_bigrams:
            if bigram not in bigrams_dist:
                bigrams_dist[bigram] = 0
                if bigram[0] in word_begin:
                    word_begin[bigram[0]] += 1
                else:
                    word_begin[bigram[0]] = 1
    print(f"test dataset complete processing")
    ##############
    # perplexity #
    ##############
    for bigram in bigrams_dist:
        if bigram[0] in word_end:
            word_end[bigram[0]] += bigrams_dist[bigram]
            # 核心思想：当有一个以word为开头的bigram时，一定有一个以word为结尾的bigram
        else:
            word_end[bigram[0]] = bigrams_dist[bigram]
    for bigram in bigrams_dist:
        bigrams_freq[bigram] = (bigrams_dist[bigram] + 1) / (word_end[bigram[0]] + word_begin[bigram[0]])
    print(f"bi-gram perplexity is {perplexity(test, bigrams_freq, 'bigram')}")


def main():
    uni_gram()
    bi_gram()


if __name__ == '__main__':
    main()
