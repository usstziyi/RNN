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
# 根据forward方法的输入输出，确定模型的输入输出
# 输入: (T, B)
# 输出: (T*B,H)
# 状态: (L*D, B, H)=(L, B, H)
class RNNModel(nn.Module):
    """循环神经网络模型"""

    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        # 循环层
        self.rnn = rnn_layer

        # 输出层
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size
        if not self.rnn.bidirectional:
            self.num_directions = 1
            # 输出层(全连接层)
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            # 输出层(全连接层)
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)

    # inputs: (T, B)
    # state:(L*D, B, H)=(L, B, H)
    def forward(self, inputs, state):
        # 1.循环层计算
        # X: (T, B, V)
        X = F.one_hot(inputs.T.long(), self.vocab_size)
        X = X.to(torch.float32)
        Y, state = self.rnn(X, state)
        # Y: (T, B, H)
        # state:(L*D, B, H)=(L, B, H)

        # 2.输出层计算
        # 全连接层首先将Y的形状改为(T*B,H)
        # 它的输出形状是(T*B,V)
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            # nn.GRU以张量作为隐状态
            # H:(L, B, H)
            return torch.zeros((self.num_directions * self.rnn.num_layers,
                                batch_size, self.num_hiddens),
                               device=device)
        else:
            # nn.LSTM以元组作为隐状态(H,C)
            # 第一个张量：隐藏状态,H:(L, B, H)
            # 第二个张量：细胞状态,C:(L, B, H)
            return (torch.zeros((
                self.num_directions * self.rnn.num_layers,
                batch_size, self.num_hiddens), device=device),
                    torch.zeros((
                        self.num_directions * self.rnn.num_layers,
                        batch_size, self.num_hiddens), device=device))


# 训练
def train_ch8(net, train_iter, vocab, lr, num_epochs, device, use_random_iter=False):
    # 定义损失函数
    loss = nn.CrossEntropyLoss()
    # 定义优化器
    updater = torch.optim.SGD(net.parameters(), lr)
    # 定义预测函数，使用 predict_ch8 函数对给定前缀进行预测，生成长度为 50 的文本
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)

    # 训练 + 预测
    for epoch in range(num_epochs):
        # ppl: 困惑度, speed: 速度（词元数量/秒）
        ppl, speed = train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter)
        print(f'epoch {(epoch + 1):3d}/{num_epochs}, 困惑度 {ppl:.1f}, {speed:.1f} 词元/秒 {str(device)}')


# 训练网络一个迭代周期，包含多个 batch,每个 batch 计算一次平均损失
# 然后根据平均损失更新一次模型参数 w 和 b
def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    """训练网络一个迭代周期（定义见第8章）"""
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2)  # 训练损失之和,词元数量
    # 处理 1 个批次数据
    for X, Y in train_iter:
        # PyTorch 内置的 GRU 模块 ：其隐藏状态 state 是一个单独的张量（形状为 [层数*方向数, 批量大小, 隐藏单元数]）， 不是元组
        # PyTorch 内置的 LSTM 模块 ：其状态是一个包含两个张量的元组 (h, c)，分别表示隐藏状态和细胞状态
        # 自定义 RNN 实现 ：通常也将状态设计为元组形式，以保持接口一致性
        if state is None or use_random_iter:
            # 在第一次迭代或使用随机抽样时初始化state
            # 使用cast确保类型检查器理解net是RNNModel实例
            state = cast(RNNModel, net).begin_state(batch_size=X.shape[0], device=device)
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                # state对于nn.GRU是个张量
                state.detach_()
            else:
                # 对于LSTM（状态是元组包含h和c）或自定义模型
                # 遍历状态中的每个张量并执行梯度分离
                for s in state:
                    s.detach_()

        # 在转置前，Y 的形状推测为 (批量大小, 时间步数)
        y = Y.T.reshape(-1)  # 转置后，形状为 (时间步数, 批量大小)，然后展平为 (时间步数 * 批量大小,)
        X, y = X.to(device), y.to(device)
        # 核心
        y_hat, state = net(X, state)  # 1.前向传播：不改变 w 和 b
        l = loss(y_hat, y.long()).mean()  # 2.计算损失：计算预测值 y_hat 与真实标签 y 之间的损失
        updater.zero_grad()  # 3.梯度清零：将 w 和 b 的梯度设为 0
        l.backward()  # 4.反向传播：计算 w 和 b 的梯度(变化方向)，不改变 w 和 b
        grad_clipping(net, 1)  # 5.梯度裁剪：将 w 和 b 的梯度裁剪到 [-1, 1] 之间
        updater.step()  # 6.更新参数: 根据梯度更新 w 和 b
        # 在没有计算 mean之前，loss 的返回值 shape 是 (时间步数 * 批量大小,)
        # 计算 mean 之后，loss 的返回值 shape 是 (1,)，即l是一个标量

        metric.add(l * y.numel(), y.numel())
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()
    # 返回困惑度、速度（词元数量/秒）


# 梯度裁切：这种裁剪方式称为 "按范数裁剪"（Clipping by Norm），是梯度裁剪中最常用的一种。
# 这相当于把所有参数的梯度拼成一个超长向量，然后计算这个向量的模长。
def grad_clipping(net, theta):  # @save
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]  # 列出所有需要计算梯度的参数
    else:
        params = net.params  # 列出所有需要计算梯度的参数

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
def predict_ch8(prefix, num_preds, net, vocab, device):
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
    """主函数：加载数据、构建模型、训练并测试"""
    # 设置超参数
    batch_size, num_steps = 32, 35  # 批量大小和时间步长
    num_hiddens = 256  # 隐藏层大小
    num_epochs, lr = 500, 1  # 训练轮数和学习率

    # 加载数据集和词汇表
    train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

    # 创建计算设备
    device = d2l.try_gpu()

    # 创建LSTM层,指定网络层数为2,双向
    lstm_layer = nn.LSTM(len(vocab), num_hiddens, num_layers=2, bidirectional=True)
    # 创建RNN模型
    net = RNNModel(lstm_layer, vocab_size=len(vocab))
    net = net.to(device)

    # 训练模型
    train_ch8(net, train_iter, vocab, lr, num_epochs, device)

    # 测试模型：预测以'time traveller'开头的10个字符
    print('--------------------------------------------------------------')
    print(predict_ch8('time traveller', 100, net, vocab, device))


if __name__ == '__main__':
    main()