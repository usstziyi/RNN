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
from typing import cast


# 继承 nn.Module
# inputs(T, B)
# outputs(T*B,H)
# state(L,B,H)
class RNNModel(nn.Module):
    """循环神经网络模型"""
    def __init__(self, num_hiddens, num_layers, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        # 循环层(D,H)
        self.rnn = nn.RNN(len(vocab), num_hiddens)
        # 输出层(H,D)
        self.linear = nn.Linear(num_hiddens, vocab_size)
            

    def begin_state(self, batch_size=1, device=None):
        if isinstance(self.rnn, nn.RNN):
            return  torch.zeros((num_layers, batch_size, num_hiddens),device=device) # state(L,B,H)
        elif isinstance(self.rnn, nn.LSTM):
            return (torch.zeros((num_layers, batch_size, num_hiddens),device=device), # state(L,B,H)
                    torch.zeros((num_layers, batch_size, num_hiddens),device=device)) # cell(L,B,H)
        else:
            raise ValueError(f'未知的RNN类型 {type(self.rnn)}')


    def forward(self, inputs, state):
        # inputs(T,B)->(T,B,D)
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
def train_rnn(net, train_iter, vocab, lr, num_epochs, device, use_random_iter=False):
    # 定义损失函数
    loss = nn.CrossEntropyLoss() # 交叉熵损失函数
    # 定义优化器
    updater = torch.optim.SGD(net.parameters(), lr)

    # 训练
    for epoch in range(num_epochs):
        timer = None, d2l.Timer()
        metric = d2l.Accumulator(2)  # 累加器，用于存储每个epoch训练损失之和和词元总数量
        # ppl: 困惑度, speed: 速度（词元数量/秒）
        ppl, speed 
        
        # 处理 1 个批次数据
        for left_mat, right_mat in train_iter:
            # 每个批次都需要初始化state
            if state is None or use_random_iter:
                state = net.begin_state(batch_size=left_mat.shape[0], device=device)
            else:
                if isinstance(net, nn.GRU):
                    # state对于nn.GRU是个张量
                    state.detach_()
                elif isinstance(net, nn.LSTM):
                    for s in state:
                        s.detach_()
                else:
                    raise ValueError(f'未知的RNN类型 {type(net.rnn)}')

            
            left_mat = left_mat.to(device)
            right_mat = right_mat.T.reshape(-1)  
            right_mat = right_mat.to(device)
            # 核心
            right_hat, state = net(left_mat, state)           # 1.前向传播：不改变 w 和 b
            l = loss(right_hat, right_mat.long()).mean()      # 2.计算损失：计算预测值 y_hat 与真实标签 y 之间的损失
            updater.zero_grad()                               # 3.梯度清零：将 w 和 b 的梯度设为 0
            l.backward()                                      # 4.反向传播：计算 w 和 b 的梯度(变化方向)，不改变 w 和 b
            grad_clipping(net, 1)                             # 5.梯度裁剪：将 w 和 b 的梯度裁剪到 [-1, 1] 之间
            updater.step()                                    # 6.更新参数: 根据梯度更新 w 和 b

            metric.add(l * right_mat.numel(), right_mat.numel())

        ppl, speed = math.exp(metric[0] / metric[1]), metric[1] / timer.stop()
        print(f'epoch {(epoch + 1):3d}/{num_epochs}, 困惑度 {ppl:.1f}, {speed:.1f} 词元/秒 {str(device)}')


# 梯度裁切：这种裁剪方式称为 "按范数裁剪"（Clipping by Norm），是梯度裁剪中最常用的一种。
# 这相当于把所有参数的梯度拼成一个超长向量，然后计算这个向量的模长。
def grad_clipping(net, theta):  #@save
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]# 列出所有需要计算梯度的参数
    else:
        params = net.params # 列出所有需要计算梯度的参数

    # 计算梯度的 L2 范数（整体梯度的"长度"）
    # 这相当于把所有参数的梯度拼成一个超长向量，然后计算这个向量的模长。
    # 只考虑有梯度的参数，避免None值导致的错误
    grad_norms = []
    for p in params:
        if p.grad is not None:
            grad_norms.append(torch.sum(p.grad ** 2))
    
    if grad_norms:  # 如果有梯度存在
        # 确保sum返回的是张量而不是标量
        total_grad_norm = torch.stack(grad_norms).sum()
        norm = torch.sqrt(total_grad_norm)
        if norm > theta:
            for param in params:
                if param.grad is not None:
                    param.grad[:] *= theta / norm
    # [:] 表示对张量内所有元素进行索引。
    # *= 是原地操作符（in-place），它不会创建新张量，而是直接修改原始张量内部的数据。
    
    # 缩放因子是 theta / norm，这样缩放后，新的梯度范数正好等于 theta。
    # 使用 param.grad[:] *= ... 是为了原地修改梯度，不影响梯度张量的内存地址（这对优化器很重要）。
    # 假设：所有参数梯度拼起来的 L2 范数是 norm = 10.0，设定阈值 theta = 5.0
    # 那么缩放因子 = 5.0 / 10.0 = 0.5，所有梯度乘以 0.5，最终范数变成 5.0。
    # 这是全局裁剪：所有参数共享同一个缩放因子，保持梯度方向不变，只缩放大小。
    # PyTorch 官方也提供了类似功能：torch.nn.utils.clip_grad_norm_，功能基本一致。

# 预测
def predict_rnn(prefix, num_preds, net, vocab, device):
    # state:(L, B, H)=(L, 1, H)
    state = net.begin_state(batch_size=1, device=device)
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
        # y: (1, H)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])


def main():
    # 设置超参数
    batch_size, num_steps = 32, 35  # 批量大小和时间步长
    num_hiddens = 256  # 隐藏层大小
    num_layers = 1  # 隐藏层数量
    num_epochs, lr = 500, 1  # 训练轮数和学习率
    device = d2l.try_gpu()
    
    # 加载数据集和词汇表
    train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
    
    
    # 创建RNN模型
    net = RNNModel(num_hiddens=num_hiddens, num_layers=num_layers, vocab_size=len(vocab))
    net = net.to(device)
    
    # 训练模型
    train_rnn(net, train_iter, vocab, lr, num_epochs, device)
    
    # 测试模型：预测以'time traveller'开头的10个字符
    print('--------------------------------------------------------------')
    print(predict_rnn('time traveller', 100, net, vocab, device))


if __name__ == '__main__':
    main()