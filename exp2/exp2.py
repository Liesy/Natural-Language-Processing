from collections import defaultdict
from typing import List
from nltk.tokenize import sent_tokenize
from snownlp import SnowNLP
from word_segmentation import *
import re


chinese = './exp2/Chinese.txt'
english = './exp2/English.txt'


def readfile(file_path) -> str:
    with open(file_path, 'r+', encoding='utf-8-sig') as f:
        data = f.read()
    return data


def seg_chinese(data: List, method: str = 'all') -> None:
    methods = ['jieba', 'snownlp', 'thulac', 'nlpir', 'stanfordcorenlp']
    result = defaultdict(list)
    result['jieba'] = method_jieba(data)
    result['snownlp'] = method_snownlp(data)
    result['thulac'] = method_thulac(data)
    result['nlpir'] = method_nlpir(data)
    result['stanfordcorenlp'] = method_stanfordcorenlp(data, 'chinese')
    if method not in methods:
        for val in result:
            print(f"segmentation result of {val} is:\n{result[val]}")
    else:
        print(f"segmentation result of {method} is:\n{result[method]}")


def seg_english(data: List, method: str = 'all') -> None:
    methods = ['nltk', 'spacy', 'stanfordcorenlp']
    result = defaultdict(list)
    result['nltk'] = method_nltk(data)
    result['spacy'] = method_spacy(data)
    result['stanfordcorenlp'] = method_stanfordcorenlp(data, 'english')
    if method not in methods:
        for val in result:
            print(f"segmentation result of {val} is:\n{result[val]}")
    else:
        print(f"segmentation result of {method} is:\n{result[method]}")


def word_seg(language: str = None) -> None:
    assert language in ['chinese', 'english']
    if language == 'chinese':
        data = readfile(chinese)
        # 分句
        sentences = re.sub('([。！？\?])([^”’])', r"\1\n\2", data).split("\n")
        seg_chinese(sentences)
    else:
        data = readfile(english)
        # 分句
        sentences = [re.sub(r'([^a-zA-Z0-9 ])', r'', sen)
                     for sen in sent_tokenize(data)]
        seg_english(sentences)


def main():
    print(f"-----Chinese-----")
    word_seg('chinese')
    print(f"\n-----English-----")
    word_seg('english')
    print(f"\n...complete processing...")


if __name__ == "__main__":
    main()
