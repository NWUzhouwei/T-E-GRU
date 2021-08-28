import numpy as np
import torch
import torch.nn as nn
import re
import jieba
import pandas as pd
from gensim.models import KeyedVectors
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from torch.utils.data import DataLoader
import warnings
import torch.optim as optim
import time
from torch.utils.tensorboard import SummaryWriter


def embedding_matrix(num_words = 50000,embedding_dim = 300):
    # 初始化embedding_matrix，之后在keras上进行应用
    embedding_matrix = np.zeros((num_words, embedding_dim))
    for i in range(num_words):
        embedding_matrix[i,:] = cn_model[ cn_model.index2word[i] ]
    embedding_matrix = embedding_matrix.astype('float32')
    return embedding_matrix

def getdata(filename, num_words=50000, max_tokens=90):
    data = pd.read_csv(filename)
    data = data.to_numpy()

    for item in data:
        text = re.sub("[\s+\/_$%^*(+\"\']+|[+——？、~@#￥%……&*（）=]+", "", item[0])
        cut = jieba.cut(text)
        cut_list = [i for i in cut]
        for i, word in enumerate(cut_list):
            try:
                cut_list[i] = cn_model.vocab[word].index
            except:
                cut_list[i] = 0
        item[0] = np.array(cut_list)

    train_pad = pad_sequences(data[:, 0], maxlen=max_tokens, padding='pre', truncating='pre')
    train_pad[train_pad >= num_words] = 0
    data_set = [(train_pad[i], data[i][1]) for i in range(len(train_pad))]

    return data_set

def test(test_db, net, batch_size=20):
    data = DataLoader(test_db, batch_size=batch_size, shuffle=True,drop_last=True)
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for text, label in data:
        with torch.no_grad():
            text = text.long().to(device)
            label = label.long().to(device)

            h = net.initHidden()
            h = h.to(device)
            output = net(text, h)
            prediction = output.topk(1)[1]

            for p, t in zip(prediction.view(-1), label.view(-1)):
                if((p==1) & (t==1)):
                    TP = TP + 1
                elif((p==0) & (t==0)):
                    TN = TN + 1
                elif((p==1) & (t==0)):
                    FP = FP +1
                elif((p==0) & (t==1)):
                    FN = FN +1
    return TP,TN,FP,FN

def getTestTime(test_db, net):
    data = DataLoader(test_db, batch_size=1, shuffle=True,drop_last=True)
    for i, (text, label) in enumerate(data):
        with torch.no_grad():
            text = text.long().to(device)
            h = net.initHidden()
            if isinstance(h,tuple):
                h, c = net.initHidden()
                h = h.to(device)
                c = c.to(device)
                h = h.to(device)
                startTestTime = time.time()
                output = net(text, h,c)
                prediction = output.topk(1)[1]
            else:
                h = h.to(device)
                startTestTime = time.time()
                output = net(text, h)
                prediction = output.topk(1)[1]
            testTime = (time.time() - startTestTime)*1000
            if (i==1):
                return testTime


def dealwithSentence(text, num_words=200000,max_tokens=100):
    text = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "", text)
    cut_list = []
    cut = jieba.cut(text)
    cut_list = [i for i in cut]
    for i, word in enumerate(cut_list):
        try:
            cut_list[i] = cn_model.vocab[word].index
        except:
            cut_list[i] = 0
    token_result = []
    token_result.append(cut_list)
    pad_result = pad_sequences(token_result, maxlen=max_tokens, padding='pre', truncating='pre')
    pad_result[pad_result >= num_words] = 0

    return torch.Tensor(pad_result).long()


print("loading  pretrain word embedding model....")
cn_model = KeyedVectors.load_word2vec_format( './embeddings/sgns.zhihu.bigram',
                                                 binary=False, unicode_errors="ignore")
if __name__ == '__main__':

    sequence_len = 400
    num_words = 150000

    T_E_hidden = 2048
    GRU_hiddenSize = 256
    head_Num = 2
    RNNDropout = 0
    T_E_dropout = 0.3

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    from GRU import Net
    print('loading  net......')
    net = Net(input_size=300, hidden_size=GRU_hiddenSize, output_size=2, batch_size=1,num_words=num_words,T_E_dropout=T_E_dropout,RNNDropout=RNNDropout,T_E_hidden=T_E_hidden,head_Num=head_Num)
    net.load_state_dict(torch.load('../log/dianping/GRU/100.pkl'))

    print('--------------------------------------------------------')
    print('Everything is OK！')
    print('--------------------------------------------------------')
    with torch.no_grad():
        while(1):
            try:
                x = dealwithSentence(input('Please Input Chinese Sentence：'),num_words,sequence_len)
                h = net.initHidden()
                output = net(x,h)
                if output.topk(1)[1] == 0:
                    print('************ Positive Sample *****************')
                else:
                    print('************ Negative Sample *****************')
            except:
                print('error，please input again!')


    # PathRoot = '../'
    # dataName = 'shopping'
    #
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #
    # Batch_size = 1
    # N_EPOCHS = 100
    # sequence_len = 200
    # num_words = 200000
    #
    # print('-------装载数据---------')
    # test_db = getdata(filename=PathRoot + 'data/' + dataName + '/test.csv', num_words=num_words,max_tokens=sequence_len)
    #
    # print('-------开始测试---------')
    # from T_E_GRU import Net
    # T_E_hidden = 2048
    # GRU_hiddenSize = 256
    # head_Num = 2
    # RNNDropout = 0
    # T_E_dropout = 0.3
    # l = [4096,2048,1024,512,256,128]
    #
    # for i in l:
    #     netName = 'T-E-GRU-' + str(T_E_hidden) + '-' + str(i)+'-'+str(T_E_dropout) + '-' + str(RNNDropout)
    #     net = Net(input_size=300, hidden_size=i, output_size=2, batch_size=Batch_size,num_words=num_words,T_E_dropout=T_E_dropout,RNNDropout=RNNDropout,T_E_hidden=T_E_hidden,head_Num=head_Num)
    #     net.load_state_dict(torch.load(PathRoot+'log/'+dataName+'/'+netName+'/100.pkl'))
    #     net = net.to(device)
    #     testTime = getTestTime(test_db,net)
    #     print(netName + ' 测试时间：' + str(testTime))
        # net = Net(input_size=300, hidden_size=256, output_size=2, batch_size=128,num_words=num_words,T_E_dropout=i,RNNDropout=i,T_E_hidden=T_E_hidden,head_Num=head_Num)
        # net.load_state_dict(torch.load(PathRoot+'log/'+dataName+'/'+netName+'/100.pkl'))
        # net = net.to(device)
        # TP, TN, FP, FN = test(test_db, net, 128)
        # accuracy = (TP + TN) / (TP + TN + FP + FN)
        # precision = TP / (TP + FP)
        # recall = TP / (TP + FN)
        # F1 = 2 * TP / (TP + TP + FP + FN)
        # print('\naccuracy:' + str(accuracy) + '\nprecision:' + str(precision) + '\nrecall:' + str(recall) + '\nF1:' + str(
        #     F1))

    # netName = 'T-E-GRU'
    # net = Net(input_size=300, hidden_size=GRU_hiddenSize, output_size=2, batch_size=Batch_size, num_words=num_words,
    #           T_E_dropout=T_E_dropout, RNNDropout=RNNDropout, T_E_hidden=2048, head_Num=head_Num)
    # net.load_state_dict(torch.load(PathRoot + 'log/' + dataName + '/' + netName + '/100.pkl'))
    # net = net.to(device)
    # testTime = getTestTime(test_db, net)
    # print(netName + ' 测试时间：' + str(testTime))


    #replace RNN

    # from T_E_RNN import Net
    # netName = 'T-E-RNN'
    # net = Net(input_size=300, hidden_size=256, output_size=2, batch_size=Batch_size,num_words=num_words)
    # net.load_state_dict(torch.load(PathRoot+'log/'+dataName+'/'+netName+'/100.pkl'))
    # net = net.to(device)
    # testTime = getTestTime(test_db,net)
    # print(netName + ' 测试时间：' + str(testTime))
    #
    # from T_E_GRU import Net
    # netName = 'T-E-GRU'
    # net = Net(input_size=300, hidden_size=256, output_size=2, batch_size=Batch_size,num_words=num_words)
    # net.load_state_dict(torch.load(PathRoot+'log/'+dataName+'/'+netName+'/100.pkl'))
    # net = net.to(device)
    # testTime = getTestTime(test_db,net)
    # print(netName + ' 测试时间：' + str(testTime))
    #
    # from T_E_LSTM import Net
    # netName = 'T-E-LSTM'
    # net = Net(input_size=300, hidden_size=256, output_size=2, batch_size=Batch_size,num_words=num_words)
    # net.load_state_dict(torch.load(PathRoot+'log/'+dataName+'/'+netName+'/100.pkl'))
    # net = net.to(device)
    # testTime = getTestTime(test_db,net)
    # print(netName + ' 测试时间：' + str(testTime))
    #
    # from T_E_BiRNN import Net
    # netName = 'T-E-BiRNN'
    # net = Net(input_size=300, hidden_size=128, output_size=2, batch_size=Batch_size,num_words=num_words)
    # net.load_state_dict(torch.load(PathRoot+'log/'+dataName+'/'+netName+'/100.pkl'))
    # net = net.to(device)
    # testTime = getTestTime(test_db,net)
    # print(netName + ' 测试时间：' + str(testTime))
    #
    # from T_E_BiLSTM import Net
    # netName = 'T-E-BiLSTM'
    # net = Net(input_size=300, hidden_size=128, output_size=2, batch_size=Batch_size,num_words=num_words)
    # net.load_state_dict(torch.load(PathRoot+'log/'+dataName+'/'+netName+'/100.pkl'))
    # net = net.to(device)
    # testTime = getTestTime(test_db,net)
    # print(netName + ' 测试时间：' + str(testTime))
    #
    # from T_E_BiGRU import Net
    # netName = 'T-E-BiGRU'
    # net = Net(input_size=300, hidden_size=128, output_size=2, batch_size=Batch_size,num_words=num_words)
    # net.load_state_dict(torch.load(PathRoot+'log/'+dataName+'/'+netName+'/100.pkl'))
    # net = net.to(device)
    # testTime = getTestTime(test_db,net)
    # print(netName + ' 测试时间：' + str(testTime))

    # test RNN with Attention

    # from RNN import Net
    # netName = 'RNN'
    # net = Net(input_size=300, hidden_size=256, output_size=2, batch_size=Batch_size,num_words=num_words)
    # net.load_state_dict(torch.load(PathRoot+'log/'+dataName+'/'+netName+'/100.pkl'))
    # net = net.to(device)
    # testTime = getTestTime(test_db,net)
    # print(netName + ' 测试时间：' + str(testTime))
    #
    # from GRU import Net
    # netName = 'GRU'
    # net = Net(input_size=300, hidden_size=256, output_size=2, batch_size=Batch_size,num_words=num_words)
    # net.load_state_dict(torch.load(PathRoot+'log/'+dataName+'/'+netName+'/100.pkl'))
    # net = net.to(device)
    # testTime = getTestTime(test_db,net)
    # print(netName + ' 测试时间：' + str(testTime))
    #
    # from LSTM import Net
    # netName = 'LSTM'
    # net = Net(input_size=300, hidden_size=256, output_size=2, batch_size=Batch_size,num_words=num_words)
    # net.load_state_dict(torch.load(PathRoot+'log/'+dataName+'/'+netName+'/100.pkl'))
    # net = net.to(device)
    # testTime = getTestTime(test_db,net)
    # print(netName + ' 测试时间：' + str(testTime))
    #
    # from Bi_RNN import Net
    # netName = 'BI_RNN'
    # net = Net(input_size=300, hidden_size=128, output_size=2, batch_size=Batch_size,num_words=num_words)
    # net.load_state_dict(torch.load(PathRoot+'log/'+dataName+'/'+netName+'/100.pkl'))
    # net = net.to(device)
    # testTime = getTestTime(test_db,net)
    # print(netName + ' 测试时间：' + str(testTime))
    #
    # from Bi_GRU import Net
    # netName = 'Bi_GRU'
    # net = Net(input_size=300, hidden_size=128, output_size=2, batch_size=Batch_size,num_words=num_words)
    # net.load_state_dict(torch.load(PathRoot+'log/'+dataName+'/'+netName+'/100.pkl'))
    # net = net.to(device)
    # testTime = getTestTime(test_db,net)
    # print(netName + ' 测试时间：' + str(testTime))
    #
    # from Bi_LSTM import Net
    # netName = 'BI_LSTM'
    # net = Net(input_size=300, hidden_size=128, output_size=2, batch_size=Batch_size,num_words=num_words)
    # net.load_state_dict(torch.load(PathRoot+'log/'+dataName+'/'+netName+'/100.pkl'))
    # net = net.to(device)
    # testTime = getTestTime(test_db,net)
    # print(netName + ' 测试时间：' + str(testTime))
    #
    # from RNN_Attention import Net
    # netName = 'RNN-Attention'
    # net = Net(input_size=300, hidden_size=256, output_size=2, batch_size=Batch_size, num_words=num_words)
    # net.load_state_dict(torch.load(PathRoot + 'log/' + dataName + '/' + netName + '/100.pkl'))
    # net = net.to(device)
    # testTime = getTestTime(test_db, net)
    # print(netName + ' 测试时间：' + str(testTime))
    #
    # from LSTM_Attention import Net
    # netName = 'LSTM-Attention'
    # net = Net(input_size=300, hidden_size=256, output_size=2, batch_size=Batch_size, num_words=num_words)
    # net.load_state_dict(torch.load(PathRoot + 'log/' + dataName + '/' + netName + '/100.pkl'))
    # net = net.to(device)
    # testTime = getTestTime(test_db, net)
    # print(netName + ' 测试时间：' + str(testTime))
    #
    # from GRU_Attention import Net
    # netName = 'GRU-Attention3'
    # net = Net(input_size=300, hidden_size=256, output_size=2, batch_size=Batch_size, num_words=num_words)
    # net.load_state_dict(torch.load(PathRoot + 'log/' + dataName + '/' + netName + '/300.pkl'))
    # net = net.to(device)
    # testTime = getTestTime(test_db, net)
    # print(netName + ' 测试时间：' + str(testTime))
    #
    # from BiRNN_Attention import Net
    # netName = 'BiRNN-Attention3'
    # net = Net(input_size=300, hidden_size=128, output_size=2, batch_size=Batch_size, num_words=num_words)
    # net.load_state_dict(torch.load(PathRoot + 'log/' + dataName + '/' + netName + '/300.pkl'))
    # net = net.to(device)
    # testTime = getTestTime(test_db, net)
    # print(netName + ' 测试时间：' + str(testTime))
    #
    # from BiGRU_Attention import Net
    # netName = 'BiGRU-Attention3'
    # net = Net(input_size=300, hidden_size=128, output_size=2, batch_size=Batch_size, num_words=num_words)
    # net.load_state_dict(torch.load(PathRoot + 'log/' + dataName + '/' + netName + '/300.pkl'))
    # net = net.to(device)
    # testTime = getTestTime(test_db, net)
    # print(netName + ' 测试时间：' + str(testTime))
    #
    # from BiLSTM_Attention import Net
    # netName = 'BiLSTM-Attention3'
    # net = Net(input_size=300, hidden_size=128, output_size=2, batch_size=Batch_size, num_words=num_words)
    # net.load_state_dict(torch.load(PathRoot + 'log/' + dataName + '/' + netName + '/300.pkl'))
    # net = net.to(device)
    # testTime = getTestTime(test_db, net)
    # print(netName + ' 测试时间：' + str(testTime))
    #
    # from Attention_RNN import Net
    # netName = 'Attention-RNN'
    # net = Net(input_size=300, hidden_size=256, output_size=2, batch_size=Batch_size, num_words=num_words)
    # net.load_state_dict(torch.load(PathRoot + 'log/' + dataName + '/' + netName + '/100.pkl'))
    # net = net.to(device)
    # testTime = getTestTime(test_db, net)
    # print(netName + ' 测试时间：' + str(testTime))
    #
    # from Attention_LSTM import Net
    # netName = 'Attention-LSTM'
    # net = Net(input_size=300, hidden_size=256, output_size=2, batch_size=Batch_size, num_words=num_words)
    # net.load_state_dict(torch.load(PathRoot + 'log/' + dataName + '/' + netName + '/100.pkl'))
    # net = net.to(device)
    # testTime = getTestTime(test_db, net)
    # print(netName + ' 测试时间：' + str(testTime))
    #
    # from Attention_GRU import Net
    # netName = 'Attention-GRU'
    # net = Net(input_size=300, hidden_size=256, output_size=2, batch_size=Batch_size, num_words=num_words)
    # net.load_state_dict(torch.load(PathRoot + 'log/' + dataName + '/' + netName + '/100.pkl'))
    # net = net.to(device)
    # testTime = getTestTime(test_db, net)
    # print(netName + ' 测试时间：' + str(testTime))
