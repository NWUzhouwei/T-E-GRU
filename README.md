# Transformer-Encoder-GRU (T-E-GRU) for Chinese Sentiment Analysis on Chinese Comment Text
## Environments
- Ubuntu 20.04
- Python 3.6.0
- Pytorch 1.7.1
 ## Prerequisites
 The code is built with following libraries (see [requirements.txt](requirements.txt)):
- tensorflow == 2.3.0
- pandas == 1.2.1
- numpy == 1.19.4
- jieba == 0.42.1
- torch == 1.7.1
- gensim == 3.8.3
```shell
 pip install -r requirements.txt
```
**Pretrain word embedding model**

> - Corpus: Zhihu_QA 知乎问答.
> - Context Features: Word + Ngram, 300Dim .
> - https://github.com/Embedding/Chinese-Word-Vectors.
> - Shen Li, Zhe Zhao, Renfen Hu, Wensi Li, Tao Liu, Xiaoyong Du, Analogical Reasoning on Chinese Morphological and Semantic Relations, ACL 2018.
```shell script
├─models                // models
│   └─embeddings        // pretrain word embedding model
│       └─sgns.zhihu.bigram.bz2
│       └─sgns.zhihu.bigram
```
## Dataset
```shell script
├─data                // Dataset
│   └─douban          // dmsc_v2
│   |   └─train.csv
│   │   └─val.csv
│   │   └─test.csv
│   └─ dianping       // yf_dianping
│   │    └─train.csv
│   │    └─val.csv
│   │    └─test.csv
│   └─shopping        // yf_amazon
│        └─train.csv
│        └─val.csv
│        └─test.csv
```
- dmsc_v2:
https://www.kaggle.com/utmhikari/doubanmovieshortcomments
- yf_dianping , yf_amazon:
http://yongfeng.me/dataset/

Specifically, about how to convert the source data to the data required by the project,
 please refer to [dealwithData.ipynb](notebooks/dealwithData.ipynb)

# Train
```shell script
├─models
│  ├─embeddings
│  │  
│  │  Attention_BiGRU.py
│  │  Attention_BiLSTM.py
│  │  Attention_BiRNN.py
│  │  Attention_GRU.py
│  │  Attention_LSTM.py
│  │  Attention_RNN.py
│  │  BiGRU_Attention.py
│  │  BiLSTM_Attention.py
│  │  BiRNN_Attention.py
│  │  Bi_GRU.py
│  │  Bi_LSTM.py
│  │  Bi_RNN.py
│  │  GRU.py
│  │  GRU_Attention.py
│  │  GRU_Attention_full.py
│  │  InformerEncoder_GRU.py
│  │  Line_Transformer_GRU.py
│  │  LSTM.py
│  │  LSTM_Attention.py
│  │  ProbAttention.py
│  │  RNN.py
│  │  RNN_Attention.py
│  │  test.py
│  │  T_E_BiGRU.py
│  │  T_E_BiLSTM.py
│  │  T_E_BiRNN.py
│  │  T_E_GRU.py
│  │  T_E_LSTM.py
│  │  T_E_RNN.py
```
***models/~.py*** files are various language models for Chinese Sentiment Analysis,
which are three types of models:
1. RNN  *(RNN, LSTM,GRU, etc)*
2. RNN with Attention  *(GRU_Attention, Attention_GRU, etc)*
3. Transformer-Encoder-RNN  *(**T_E_GRU**, T_E_RNN, T_E_LSTM, etc)*
---
To train these models, you need :
- Modify hyper-parameters or dataName;
```python
if __name__ == '__main__':
    # You can modify the value of each variable before the net is instantiated
```
- Just run following:
```shell script
python xxxx.py 
```
**Note**:
- If your GPU is available, it will run to accelerate training, otherwise it will only be CPU by default;
- During training, it will provide rough estimations of accuracies and loss;
- Test function  also in training file, was name ***test()***;
- After the training, the training log and the final model will be generated in the ***log*** folder, for example
```shell script
├─log
│  ├─douban
│  │  ├─T_E_GRU 
│  │  │  │  100.pkl
│  │  │  │  92.pkl
│  │  │  │  94.pkl
│  │  │  │  96.pkl
│  │  │  │  98.pkl
│  │  │  │  测试结果.txt  # result on testSet
│  │  │  ├─Acc_test
│  │  │  ├─Acc_train
│  │  │  ├─Loss_test
│  │  │  └─Loss_train

# if you want to know more training details, you can use the following command:
tensorboard --logdir=log
```

# Test
You can also write a simple script to test or use the trained models as ***[./models/test.py](./models/test.py)***
```shell script
├─models
│  ├─test.py
```

## Notebook
- ***./notebook*** has some detail about data processing and model training.
- If you want to run them, **Jupyter lab** is the best.;

## Performance

Since there are many models involved in this project, only **Transformer-Encoder-GRU (T-E-GRU)** can achieve performance in 100 epochs on each data set, as shown in the table：

|     DataSet      | Accuracy |   F1   | Test Time(a sample) |
| :--------------: | :------: | :----: | :-----------------: |
|     DMSC\_V2     |  90.09%  | 90.07% |      2.6887ms       |
|   YF\_DianPing   |  90.76%  | 90.98% |      8.0953ms       |
| Online\_shopping |  92.92%  | 93.05% |      4.3921ms       |

