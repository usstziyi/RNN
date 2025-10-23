import matplotlib.pyplot as plt

# 初始化绘图
def init_plot(lr):
    plt.ion()  # 开启交互模式
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs_list = [] # x轴数据
    ppls_list = [] # y轴数据
    line, = ax.plot(epochs_list, ppls_list, 'b-', linewidth=2, label='Perplexity')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Perplexity')
    ax.set_title(f'RNN Training Perplexity vs Epoch (lr={lr})')
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    return fig, ax, line, epochs_list, ppls_list
    # fig是画布
    # ax是坐标系区域(axis是带刻度的图表框)，一个fig 上可以有多个 ax
    # line是线对象
    # epochs_list和ppls_list是数据列表


# 更新绘图
def update_plot(epoch, ppl, epochs_list, ppls_list, line, ax):
    epochs_list.append(epoch)
    ppls_list.append(ppl)
    line.set_xdata(epochs_list)
    line.set_ydata(ppls_list)
    ax.set_xlim(0, epoch + 2)  # 确保x轴范围包含当前epoch，右边预留2个单位
    ax.set_ylim(0, max(ppls_list) * 1.1 if ppls_list else 1)  # 防止空列表报错, y轴预留10%
    plt.draw()
    plt.pause(0.01)

# 关闭绘图
def close_plot():
    plt.ioff()  # 关闭交互模式->恢复默认行为
    plt.show()  # 阻塞，保持窗口打开直到用户手动关闭