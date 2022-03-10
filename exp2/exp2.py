from collections import defaultdict
from typing import List
from nltk.tokenize import sent_tokenize
import re, jieba


chinese = './exp2/Chinese.txt'
english = './exp2/English.txt'


def readfile(file_path) -> str:
    with open(file_path, 'r+', encoding='utf-8') as f:
        data = f.read()
    return data


def seg_chinese(data: List, method: str = 'all') -> None:
    methods = ['jieba', 'snownlp', 'thulac', 'nlpir', 'stanfordcorenlp']
    result = defaultdict(list)
    if method not in methods:
        for key, val in result:
            print(f"segmentation result of {key} is: {val}")
    else:
        print(f"segmentation result of {method} is: {result[method]}")


def seg_english(data: List, method: str = 'all') -> None:
    methods = ['nltk', 'spacy', 'stanfordcorenlp']
    result = defaultdict(list)
    if method not in methods:
        for key, val in result:
            print(f"segmentation result of {key} is: {val}")
    else:
        print(f"segmentation result of {method} is: {result[method]}")


def word_seg(language: str = None) -> None:
    assert language in ['chinese', 'english']
    if language == 'chinese':
        data = readfile(chinese)
        # 分句，对每句处理标点符号
        sentences = re.sub('([。！？\?])([^”’])', r"\1\n\2", data).split("\n")
        print(sentences)
        exit(0)
        seg_chinese(data)
    else:
        data = readfile(english)
        # 分句，对每句删掉标点符号
        sentences = [re.sub(r'[^a-zA-Z0-9 ]', r'', sen) for sen in sent_tokenize(data)]
        seg_english(sentences)


def main():
    print(f"-----Chinese-----")
    word_seg('chinese')
    print(f"\n-----English-----")
    word_seg('english')
    print(f"\n...complete processing...")


if __name__ == "__main__":
    main()
