import numpy as np
import torch
import torch.nn as nn
import re
import jieba
import pandas as pd
from gensim.models import KeyedVectors
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from torch.utils.data import random_split,DataLoader
import torch.nn.functional as F
import warnings
import torch.optim as optim
import time
from torch.utils.tensorboard import SummaryWriter
from math import sqrt
warnings.filterwarnings("ignore")


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
def embedding_matrix(num_words = 50000,embedding_dim = 300):

    # 初始化embedding_matrix，之后在keras上进行应用
    embedding_matrix = np.zeros((num_words, embedding_dim))
    for i in range(num_words):
        embedding_matrix[i,:] = cn_model[ cn_model.index2word[i] ]
    embedding_matrix = embedding_matrix.astype('float32')
    return embedding_matrix


class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dytpe=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask
class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask
class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):
        # Q [B, H, L, D]
        B, H, L, E = K.shape
        _, _, S, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, S, L, E)
        indx_sample = torch.randint(L, (S, sample_k))
        K_sample = K_expand[:, :, torch.arange(S).unsqueeze(1), indx_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            V_sum = V.sum(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            assert (L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-1)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V)
        return context_in

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, D = queries.shape
        _, S, _, _ = keys.shape

        queries = queries.view(B, H, L, -1)
        keys = keys.view(B, H, S, -1)
        values = values.view(B, H, S, -1)

        U = self.factor * torch.ceil(torch.log(torch.Tensor([S]))).int().item()
        u = self.factor * torch.ceil(torch.log(torch.Tensor([L]))).int().item()

        scores_top, index = self._prob_QK(queries, keys, u, U)
        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L)
        # update the context with selected top_k queries
        context = self._update_context(context, values, scores_top, index, L, attn_mask)

        return context.contiguous()
class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        ).view(B, L, -1)

        return self.out_projection(out)
class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x
class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        x = x + self.dropout(self.attention(
            x, x, x,
            attn_mask=attn_mask
        ))

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y)
class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
            x = self.attn_layers[-1](x)
        else:
            for attn_layer in self.attn_layers:
                x = attn_layer(x, attn_mask=attn_mask)

        if self.norm is not None:
            x = self.norm(x)

        return x

class Net(nn.Module):
    def __init__(self, input_size=300, dropout=0.05, factor=5, n_heads=2, d_ff=1024, e_layers=2, batch_size=32,
                 num_layers=1, hidden_size=128, output_size=2,num_words=50000):
        super().__init__()
        self.d_model = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size

        self.embedding = nn.Embedding.from_pretrained(torch.Tensor(embedding_matrix(num_words=num_words)))
        self.embedding.requires_grad = False


        self.batch_size = batch_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(ProbAttention(False, factor, attention_dropout=dropout),
                                   input_size, n_heads),
                    input_size,
                    d_ff,
                    dropout=dropout,
                    activation='gelu'
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    input_size
                ) for l in range(e_layers - 1)
            ],
            norm_layer=torch.nn.LayerNorm(input_size)
        )

        self.rnn = nn.RNN(input_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, h, atten_mask=None):
        x = self.embedding(x)

        x = self.encoder(x)
        x = x.transpose(0, 1)
        x, h = self.rnn(x, h)
        x = self.linear(x[-1])
        x = self.softmax(x)
        return x

    def initHidden(self):
        h_0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
        return h_0


def train(train_db, net, batch_size=20):
    train_loss = 0
    train_acc = 0

    data = DataLoader(train_db, batch_size=batch_size, shuffle=True,drop_last=True)

    for i, (text, label) in enumerate(data):
        optimizer.zero_grad()

        text = text.long().to(device)
        label = label.long().to(device)

        h = net.initHidden()
        h = h.to(device)
        output = net(text, h)
        loss = criterion(output, label)

        train_acc += (label.view(-1, 1) == output.topk(1)[1]).sum().item()
        train_loss += loss.item()

        loss.backward()
        optimizer.step()

    return train_loss / len(train_db), train_acc / len(train_db)


def valid(val_db, net, batch_size=20):
    val_loss = 0
    val_acc = 0

    data = DataLoader(val_db, batch_size=batch_size, shuffle=True,drop_last=True)

    for text, label in data:
        with torch.no_grad():
            text = text.long().to(device)
            label = label.long().to(device)

            h = net.initHidden()
            h = h.to(device)
            output = net(text, h)
            loss = criterion(output, label)

            val_acc += (label.view(-1, 1) == output.topk(1)[1]).sum().item()
            val_loss += loss.item()

    return val_loss / len(val_db), val_acc / len(val_db)


if __name__ == '__main__':
    PathRoot = '../'
    netName = 'InformerEncoder_GRU'
    writer = SummaryWriter(log_dir=PathRoot +'log/'+ netName,comment="InformerEncoder_GRU,lr=0.01，batch_size=64,layer=1,embedding=200000")

    print('-------加载词嵌入模型---------')
    # 使用gensim加载预训练中文分词embedding, 有可能需要等待1-2分钟
    cn_model = KeyedVectors.load_word2vec_format( './embeddings/sgns.zhihu.bigram',
                                                 binary=False, unicode_errors="ignore")

    print('-------初始化网络---------')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Batch_size = 128
    N_EPOCHS = 100
    sequence_len = 100
    num_words = 200000
    input_size = 300
    dropout = 0.05
    factor = 5
    n_heads = 2
    d_ff = 1024  # 全链接层的隐层
    e_layers = 2

    net = Net(input_size=input_size, dropout=dropout, factor=factor, n_heads=n_heads, d_ff=d_ff, e_layers=e_layers,batch_size=Batch_size,num_words=num_words)
    # 装载之前训练的部分继续训练
    # net.load_state_dict(torch.load(PathRoot+'models/'+netName+'.pkl'))

    net = net.to(device)
    criterion = nn.NLLLoss().to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    writer.add_graph(net, (torch.zeros(Batch_size, sequence_len).long().to(device), net.initHidden().to(device)))
    print('网络结构为：')
    print(net)

    print('-------加载数据---------')
    train_db = getdata(filename=PathRoot+'data/trainData_60w.csv', num_words=num_words, max_tokens=sequence_len)
    val_db = getdata(filename=PathRoot+'data/testData_10w.csv', num_words=num_words, max_tokens=sequence_len)
    print('-------训练开始---------')
    print('-------运算设备为：' + str(device) + '---------')

    start_time = time.time()
    for epoch in range(N_EPOCHS):

        train_loss, train_acc = train(train_db, net, Batch_size)
        valid_loss, valid_acc = valid(val_db, net, Batch_size)
        scheduler.step()

        secs = int(time.time() - start_time)

        mins = secs / 60
        secs = secs % 60

        writer.add_scalars('Loss', {'train': train_loss,'test': valid_loss}, epoch)
        writer.add_scalars('Acc', {'train': train_acc,'test': valid_acc}, epoch)

        print('Epoch: %d' % (epoch + 1), " | time in %d minites, %d seconds" % (mins, secs))
        print(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)')
        print(f'\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)')

    torch.save(net.state_dict(), '../log/'+netName+'.pkl')

    writer.close()