# Natural Language Processing

**Star it if it's useful for you! Thanks!**

### Author

- Yang Li 李 阳
- Artificial Intelligence Class of 2019
- School of Computer Science and Technology
- Shandong University

### Change Log

- 2022-3-6 init commits

### Experiment Content

[submit](https://icloud.qd.sdu.edu.cn:7777/link/928B36E8072A8C857687200257746BE7)

#### exp1 语言模型

1. 用python编程实践语言模型（uni-gram和bi-gram）,加入平滑技术。
2. 计算test.txt中句子的PPL，对比uni-gram和bi-gram语言模型效果。
3. 数据集
   > train_LM.txt test_LM.txt
   >
   > Example：（每行数据是一段对话，句子间用\_\_eou\_\_分隔）
   >
   > How much can I change 100 dollars for ? \_\_eou\_\_ What kind of currency do you want ? \_\_eou\_\_ How much will it be in Chinese currency ? \_\_eou\_\_ That's 680 Yuan . \_\_eou\_\_
   >
   > What kind of account do you prefer ? Checking account or savings account ? \_\_eou\_\_ I would like to open a checking account . \_\_eou\_\_Ok , please just fill out this form and show us your ID card .\_\_eou\_\_ Here you are . \_\_eou\_\_``

#### exp2 分词

1. 利用给定的中英文文本序列(Chinese.txt and English.txt)，分别利用以下给定的中英文分词工具进行分词并对不同分词工具产生的结果进行简要对比分析。
2. 中文分词工具
   - Jieba(重点)，尝试三种分词模式与自定义词典功能
   - SnowNLP
   - THULAC
   - NLPIR
   - StanfordCoreNLP
3. 英文分词工具
   - NLTK
   - SpaCy
   - StanfordCoreNLP

#### exp3 词性标注

1. 利用 Chinese.txt 和 English.txt 的中英文句子，在实验二的基础上，继续利用以下给定的中英文工具进行词性标注。并对不同工具产生的结果进行简要对比分析。

2. 使用python编程实践CRF，进行词性标注。该实验基于python3.6以及keras训练bi-lstm,结合CRF来实现词性标注。

#### exp4 命名实体识别

1. 利用 Chinese.txt 和 English.txt 的中英文句子，在实验二的基础上，继续利用以下给定的中英文工具进行命名实体识别。

2. 使用BERT + Bi-LSTM + CRF 实践命名实体识别。

> 1. model parameters
> 
> 在./experiments/clue/config.json中设置了Bert模型的基本参数，
> 而在./pretrained\_bert\_models下的预训练文件夹中，config.json除了设置Bert的基本参数外，
> 还设置了LSTM参数，可根据需要进行更改。
> 
> 2. other parameters
> 
> 环境路径以及其他超参数在./config.py中进行设置。
>
> 3. run
>
> python run.py
>
> 模型运行结束后，最优模型和训练log保存在./experiments/clue/路径下。在测试集中的bad case保存在./case/bad\_case.txt中。
> 如要重新运行模型，请先将train.log移出当前路径，以免覆盖。

#### exp5 词向量

1. OneHot编码。```从one-hot编码结果来看，one-hot编码的缺点是什么？```

2. Word2vec词向量训练。