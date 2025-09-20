"""
循环神经网络(RNN)实现 - 基于PyTorch
此代码实现了一个基本的循环神经网络模型，用于文本预测任务。
使用d2l库提供的工具函数加载《时间机器》数据集，并训练模型进行字符级预测。
"""

# 导入必要的库
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
import math


# 继承 nn.Module
# inputs(T, B)
# outputs(T*B,H)
# state(L,B,H)
class RNNModel(nn.Module):
    """循环神经网络模型"""
    def __init__(self, hidden_size, num_layers, vocab, **kwargs):
        super(RNNModel, self).__init__(**kwargs)

        self.vocab_size = len(vocab)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        


        # 循环层(D,H)
        self.rnn = nn.RNN(
            input_size = self.vocab_size, 
            hidden_size = self.hidden_size, 
            num_layers = self.num_layers,
            nonlinearity='tanh',
            bias=True,
            batch_first=False,
            dropout=0,
            bidirectional=False,
        )
        # self.rnn = nn.GRU(
        #     input_size = self.vocab_size, 
        #     hidden_size = self.hidden_size, 
        #     num_layers = self.num_layers
        # )
        # self.rnn = nn.LSTM(
        #     input_size = self.vocab_size, 
        #     hidden_size = self.hidden_size, 
        #     num_layers = self.num_layers
        # )
        # 输出层(H,D)
        self.linear = nn.Linear(self.hidden_size, self.vocab_size)

    # 给 nn.RNN 或 nn.LSTM 初始化隐藏状态
    def begin_state(self, batch_size=1, device=None):
        if isinstance(self.rnn, nn.RNN):
            return torch.zeros((self.num_layers, batch_size, self.hidden_size),device=device) # state(L,B,H)
        elif isinstance(self.rnn, nn.GRU):
            return torch.zeros((self.num_layers, batch_size, self.hidden_size),device=device) # state(L,B,H)
        elif isinstance(self.rnn, nn.LSTM):
            return (torch.zeros((self.num_layers, batch_size, self.hidden_size),device=device), # state(L,B,H)
                    torch.zeros((self.num_layers, batch_size, self.hidden_size),device=device)) # cell(L,B,H)
        else:
            raise ValueError(f'未知的RNN类型 {type(self.rnn)}')

    # inputs(B,T)
    # state(L,B,H)
    # outputs(T*B,D)
    def forward(self, inputs, state):
        # inputs(B,T)->(T,B)->(T,B,D)
        inputs = F.one_hot(inputs.T.long(), self.vocab_size).to(torch.float32) # 独热编码
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

# 训练
def train_rnn(net, train_iter, lr, num_epochs, device, use_random_iter=False):
    # 定义损失函数
    loss = nn.CrossEntropyLoss(reduction='none') # 交叉熵损失函数
    # 定义优化器
    updater = torch.optim.SGD(net.parameters(), lr)

    # 训练
    for epoch in range(num_epochs):
        timer = d2l.Timer()
        metric = d2l.Accumulator(2)  # 累加器，用于存储每个epoch训练损失之和和词元总数量 
        
        state = None       
        # left_mat(B,T)
        # right_mat(B,T)
        for left_mat, right_mat in train_iter:

            # 初始化state(L,B,H)
            if state is None or use_random_iter:
                state = net.begin_state(batch_size=left_mat.shape[0], device=device)
            else: 
                if isinstance(net, nn.Module) and not isinstance(state, tuple):
                    state.detach_()
                else:
                    for s in state:
                        s.detach_()

            # 整理数据
            # left_mat(B,T)
            left_mat = left_mat.to(device)
            right_mat = right_mat.T.reshape(-1) 
            # right_mat(T*B,)
            right_mat = right_mat.to(device)


            # 训练核心
            # left_mat(B,T)
            # state(L,B,H)
            # right_fore(T*B,D)
            # state(L,B,H)
            right_fore, state = net(left_mat, state)           # 1.前向传播：不改变 w 和 b
            # right_fore(T*B,D), right_mat(T*B,)
            # loss(T*B,)
            # l(1)
            l = loss(right_fore, right_mat.long()).mean()      # 2.计算损失：计算预测值 y_fore 与真实标签 y 之间的损失
            updater.zero_grad()                                # 3.梯度清零：将 w 和 b 的梯度设为 0
            l.backward()                                       # 4.反向传播：计算 w 和 b 的梯度(变化方向)，不改变 w 和 b
            grad_clipping(net, 1)                              # 5.梯度裁剪：将 w 和 b 的梯度裁剪到 [-1, 1] 之间
            updater.step()                                     # 6.更新参数: 根据梯度更新 w 和 b
            
            # 累加损失，累加词元总数
            metric.add(l * right_mat.numel(), right_mat.numel())

        ppl, speed = math.exp(metric[0] / metric[1]), metric[1] / timer.stop()
        print(f'epoch {(epoch + 1):3d}/{num_epochs}, 困惑度 {ppl:.1f}, {speed:.1f} 词元/秒 {str(device)}')


# 梯度裁切：为了防止梯度爆炸，我们可以对梯度进行裁剪，将其限制在一个指定的范围内。
# 这种裁剪方式称为 "按范数裁剪"（Clipping by Norm），是梯度裁剪中最常用的一种。
# 这相当于把所有参数的梯度拼成一个超长向量，然后计算这个向量的模长。
# theta 是裁剪阈值，用于限制梯度的范数。
def grad_clipping(net, theta):
    # 列出所有需要计算梯度的参数
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params

    # 计算梯度的 L2 范数（整体梯度的"长度"）
    # 只考虑有梯度的参数，避免None值导致的错误
    grad_norms = []
    for p in params:
        if p.grad is not None:
            grad_norms.append(torch.sum(p.grad ** 2))
    
    if grad_norms:  # 如果有梯度存在
        # 确保sum返回的是张量而不是标量
        # torch.stack 的输入必须是张量，输出也一定是张量
        total_grad_norm = torch.stack(grad_norms).sum()
        norm = torch.sqrt(total_grad_norm)
        if norm > theta:
            for param in params:
                if param.grad is not None:
                    param.grad[:] *= theta / norm
    # param.grad *= theta / norm 本身也是原地操作
    # param.grad[:] *= theta / norm 也是原地操作
    
    # 缩放因子是 theta / norm，这样缩放后，新的梯度范数正好等于 theta。
    # 使用 param.grad[:] *= ... 是为了原地修改梯度，不影响梯度张量的内存地址（这对优化器很重要）。
    # 假设：所有参数梯度拼起来的 L2 范数是 norm = 10.0，设定阈值 theta = 5.0
    # 那么缩放因子 = 5.0 / 10.0 = 0.5，所有梯度乘以 0.5，最终范数变成 5.0。
    # 这是全局裁剪：所有参数共享同一个缩放因子，保持梯度方向不变，只缩放大小。
    # PyTorch 官方也提供了类似功能：torch.nn.utils.clip_grad_norm_，功能基本一致。

# 预测
# prefix(str): 提供的字符串
# num_preds:需要向后预测多少位
def predict_rnn(prefix, num_preds, net, vocab, device):
    # state:(L, B, H)=(L, 1, H)
    state = net.begin_state(batch_size=1, device=device)
    # outputs(list)
    outputs = [vocab[prefix[0]]]
    # get_input:(T, B)=(1, 1)
    get_input = lambda: d2l.reshape(d2l.tensor([outputs[-1]], device=device), (1, 1))
    # 预热，优化state
    for y in prefix[1:]:
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    #  预测, 每次预测一个字符
    for _ in range(num_preds):
        y, state = net(get_input(), state)
        # y: (1, H)->(1)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])


def main():
    # 设置超参数
    batch_size = 32     # B:批量大小
    num_steps = 35      # T:时间步长
    hidden_size = 256   # H:隐藏层大小
    num_layers = 1      # L:隐藏层数量
    num_epochs, lr = 500, 1  # 训练轮数和学习率
    device = d2l.try_gpu()
    
    # 加载数据集和词汇表
    train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
    
    
    # 创建RNN模型
    net = RNNModel(hidden_size, num_layers, vocab)
    net = net.to(device)
    
    # 训练模型
    train_rnn(net, train_iter, lr, num_epochs, device)
    
    # 测试模型：预测以'time traveller'开头的10个字符
    print('--------------------------------------------------------------')
    print(predict_rnn('time traveller', 100, net, vocab, device))


if __name__ == '__main__':
    main()