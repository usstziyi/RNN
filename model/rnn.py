import torch
from torch import nn
from torch.nn import functional as F


# 继承 nn.Module
# inputs(T, B)
# outputs(T*B,H)
# state(L,B,H)
class RNNModel(nn.Module):
    """循环神经网络模型"""
    def __init__(self, hidden_size, num_layers, vocab, **kwargs):
        super(RNNModel, self).__init__(**kwargs)

 
        


        # 循环层(D,H)
        self.rnn = nn.RNN(
            input_size = len(vocab),
            hidden_size = hidden_size, 
            num_layers = num_layers,
            nonlinearity='tanh',
            bias=True,
            batch_first=False,
            dropout=0,
            bidirectional=False,
        )
        # self.rnn = nn.GRU(
        #     input_size = len(vocab), 
        #     hidden_size = self.hidden_size, 
        #     num_layers = self.num_layers
        # )
        # self.rnn = nn.LSTM(
        #     input_size = len(vocab), 
        #     hidden_size = self.hidden_size, 
        #     num_layers = self.num_layers
        # )
        # 输出层(H,D)
        self.linear = nn.Linear(
            in_features=self.rnn.hidden_size, 
            out_features=self.rnn.input_size
        )

    # 给 nn.RNN 或 nn.LSTM 初始化隐藏状态
    def begin_state(self, batch_size=1, device=None):
        if isinstance(self.rnn, nn.RNN):
            return torch.zeros((self.rnn.num_layers, batch_size, self.rnn.hidden_size),device=device) # state(L,B,H)
        elif isinstance(self.rnn, nn.GRU):
            return torch.zeros((self.rnn.num_layers, batch_size, self.rnn.hidden_size),device=device) # state(L,B,H)
        elif isinstance(self.rnn, nn.LSTM):
            return (torch.zeros((self.rnn.num_layers, batch_size, self.rnn.hidden_size),device=device), # state(L,B,H)
                    torch.zeros((self.rnn.num_layers, batch_size, self.rnn.hidden_size),device=device)) # cell(L,B,H)
        else:
            raise ValueError(f'未知的RNN类型 {type(self.rnn)}')

    # inputs(B,T)
    # state(L,B,H)
    # outputs(T*B,D)
    def forward(self, inputs, state):
        # inputs(B,T)->(T,B)->(T,B,D)
        inputs = F.one_hot(inputs.T.long(), self.rnn.input_size).to(torch.float32) # 独热编码
        # inputs(T,B,D)
        # state(L,B,H)
        # rnn(D,H)
        # states(T,B,H)
        # state(L,B,H)
        states, state = self.rnn(inputs, state)
        # states是所有时间步的隐藏状态
        # state是最后一个时间步的所有隐藏状态

        # states(T,B,H)->(T*B,H)
        # linear(H,D)
        # outputs(T*B,D)
        output = self.linear(states.reshape((-1, states.shape[-1])))
        return output, state