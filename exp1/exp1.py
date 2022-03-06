from nltk.tokenize import word_tokenize
import re


def readfile(file_path):
    dialogues = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            dialogue = [re.sub(r'[^a-zA-Z0-9 ]', r'', text).strip().lower() for text in line.split(sep='__eou__')]
            dialogues.append(dialogue)

    return dialogues


def main():
    train = readfile('train_LM.txt')
    for sentence in train[1]:
        print(sentence)


if __name__ == '__main__':
    main()
