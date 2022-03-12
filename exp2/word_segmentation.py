from typing import List
import jieba
import snownlp
import thulac
from stanfordcorenlp import StanfordCoreNLP
import nltk
import spacy
import pynlpir as nlpir


stanford_path = r'/home/liyang/cv/nlp/exp2/stanford-corenlp-full-2016-10-31/'
spacy_path = 'en_core_web_sm'


def method_jieba(sentences: List) -> List:
    word_list = []
    for sen in sentences:
        words = jieba.cut(sen)
        for word in words:
            word_list.append(word)
    return word_list


def method_snownlp(sentences: List) -> List:
    word_list = []
    for sen in sentences:
        words = snownlp.SnowNLP(sen).words
        for word in words:
            word_list.append(word)
    return word_list


def method_thulac(sentences: List) -> List:
    thu = thulac.thulac(seg_only=True)  # 不进行词性标注
    word_list = []
    for sen in sentences:
        words = thu.cut(sen)
        for word in words:
            word_list.append(word[0])
    return word_list


def method_nlpir(sentences: List) -> List:
    nlpir.open()
    word_list = []
    for sen in sentences:
        words = nlpir.segment(sen, pos_tagging=False)
        for word in words:
            word_list.append(word)
    return word_list


def method_stanfordcorenlp(sentences: List, language: str = None) -> List:
    assert language in ['chinese', 'english']
    word_list = []
    stanford = StanfordCoreNLP(stanford_path, lang='zh') if language == 'chinese' else StanfordCoreNLP(stanford_path)
    for sen in sentences:
        words = stanford.word_tokenize(sentences)
        for word in words:
            word_list.append(word)
    return word_list


def method_nltk(sentences: List) -> List:
    word_list = []
    for sen in sentences:
        words = nltk.tokenize.word_tokenize(sen)
        for word in words:
            word_list.append(word)
    return word_list


def method_spacy(sentences: List) -> List:
    class WhitespaceTokenizer(object):
        def __init__(self, vocab):
            self.vocab = vocab

        def __call__(self, text):
            words = text.split(' ')
            # All tokens 'own' a subsequent space character in this tokenizer
            spaces = [True] * len(words)
            return spacy.tokens.Doc(self.vocab, words=words, spaces=spaces)
    model = spacy.load(spacy_path)
    model.tokenizer = WhitespaceTokenizer(model.vocab)
    word_list = []
    for sen in sentences:
        words = model(sen)
        for word in words:
            word_list.append(word)
    return word_list
