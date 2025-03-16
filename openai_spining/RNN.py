import numpy as np


class CharRNN:
    def __init__(self, chars, hidden_size=100, seq_length=25, learning_rate=1e-1):
        """
        初始化字符级RNN语言模型

        参数:
        - chars: 字符集合，用于建立字符到索引的映射
        - hidden_size: 隐藏层大小
        - seq_length: 训练时使用的序列长度
        - learning_rate: 学习率
        """
        self.chars = chars
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.learning_rate = learning_rate

        # 建立字符到索引的映射
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)

        # 初始化模型参数
        self.init_parameters()

    def init_parameters(self):
        """初始化权重和偏置"""
        # 输入到隐藏层的权重
        self.Wxh = np.random.randn(self.hidden_size, self.vocab_size) * 0.01
        # 隐藏层到隐藏层的权重
        self.Whh = np.random.randn(self.hidden_size, self.hidden_size) * 0.01
        # 隐藏层到输出层的权重
        self.Why = np.random.randn(self.vocab_size, self.hidden_size) * 0.01
        # 偏置项
        self.bh = np.zeros((self.hidden_size, 1))
        self.by = np.zeros((self.vocab_size, 1))

        # 参数梯度的内存
        self.mWxh, self.mWhh, self.mWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        self.mbh, self.mby = np.zeros_like(self.bh), np.zeros_like(self.by)

    def forward(self, inputs, hprev):
        """
        前向传播

        参数:
        - inputs: 输入字符的索引列表
        - hprev: 初始隐藏状态

        返回:
        - 模型输出和状态信息的字典
        """
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = np.copy(hprev)
        loss = 0

        # 前向传播
        for t in range(len(inputs)):
            # 将输入字符编码为one-hot向量
            xs[t] = np.zeros((self.vocab_size, 1))
            xs[t][inputs[t]] = 1

            # 计算隐藏状态
            hs[t] = np.tanh(np.dot(self.Wxh, xs[t]) + np.dot(self.Whh, hs[t - 1]) + self.bh)

            # 计算输出层
            ys[t] = np.dot(self.Why, hs[t]) + self.by

            # softmax处理得到概率分布
            ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))

            # 计算交叉熵损失
            loss += -np.log(ps[t][inputs[(t + 1) % len(inputs)], 0])

        # 返回全部状态信息
        return xs, hs, ys, ps, loss

    def backward(self, inputs, xs, hs, ps):
        """
        反向传播

        参数:
        - inputs: 输入字符的索引列表
        - xs, hs, ps: 前向传播得到的状态

        返回:
        - 参数梯度和最终隐藏状态
        """
        # 初始化梯度
        dWxh, dWhh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
        dhnext = np.zeros_like(hs[0])

        # 反向传播
        for t in reversed(range(len(inputs))):
            # 输出层的梯度
            dy = np.copy(ps[t])
            # 对目标字符的梯度进行调整
            dy[inputs[(t + 1) % len(inputs)]] -= 1

            # 更新隐藏层到输出层的权重梯度
            dWhy += np.dot(dy, hs[t].T)
            dby += dy

            # 计算隐藏层梯度
            dh = np.dot(self.Why.T, dy) + dhnext
            # tanh的导数: (1 - tanh^2)
            dhraw = (1 - hs[t] * hs[t]) * dh

            # 更新偏置梯度
            dbh += dhraw

            # 更新权重梯度
            dWxh += np.dot(dhraw, xs[t].T)
            dWhh += np.dot(dhraw, hs[t - 1].T)

            # 下一个时间步的隐藏状态梯度
            dhnext = np.dot(self.Whh.T, dhraw)

        # 梯度裁剪，防止梯度爆炸
        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)

        return dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs) - 1]

    def update_parameters(self, dWxh, dWhh, dWhy, dbh, dby):
        """
        使用Adagrad更新参数

        参数:
        - dWxh, dWhh, dWhy, dbh, dby: 参数梯度
        """
        # 实现Adagrad优化算法
        for param, dparam, mem in zip(
                [self.Wxh, self.Whh, self.Why, self.bh, self.by],
                [dWxh, dWhh, dWhy, dbh, dby],
                [self.mWxh, self.mWhh, self.mWhy, self.mbh, self.mby]
        ):
            mem += dparam * dparam
            param -= self.learning_rate * dparam / np.sqrt(mem + 1e-8)

    def sample(self, h, seed_idx, n):
        """
        使用模型采样新文本

        参数:
        - h: 初始隐藏状态
        - seed_idx: 种子字符的索引
        - n: 要生成的字符数

        返回:
        - 生成的字符串
        """
        x = np.zeros((self.vocab_size, 1))
        x[seed_idx] = 1
        generated_chars = []

        for t in range(n):
            # 前向传播一步
            h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
            y = np.dot(self.Why, h) + self.by
            p = np.exp(y) / np.sum(np.exp(y))

            # 根据概率采样
            idx = np.random.choice(range(self.vocab_size), p=p.ravel())

            # 更新输入
            x = np.zeros((self.vocab_size, 1))
            x[idx] = 1

            generated_chars.append(self.idx_to_char[idx])

        return ''.join(generated_chars)

    def train(self, data, iterations=100, print_interval=100):
        """
        训练模型

        参数:
        - data: 训练文本
        - iterations: 迭代次数
        - print_interval: 打印间隔
        """
        # 数据预处理
        data_indices = [self.char_to_idx[ch] for ch in data]
        n_samples = len(data_indices) - self.seq_length

        # 初始隐藏状态
        h = np.zeros((self.hidden_size, 1))

        # 训练循环
        for i in range(iterations):
            # 随机选择起始点
            start_idx = np.random.randint(0, n_samples)
            inputs = data_indices[start_idx:start_idx + self.seq_length]

            # 前向传播
            xs, hs, ys, ps, loss = self.forward(inputs, h)

            # 反向传播
            dWxh, dWhh, dWhy, dbh, dby, h = self.backward(inputs, xs, hs, ps)

            # 更新参数
            self.update_parameters(dWxh, dWhh, dWhy, dbh, dby)

            # 打印损失和样本
            if i % print_interval == 0:
                print(f'Iteration {i}, loss: {loss}')
                sample = self.sample(h, inputs[0], 200)
                print(f'Sample: {sample}\n')


# 简单的使用示例
def example_usage():
    # 加载文本数据
    data = open('text_corpus.txt', 'r').read()
    chars = sorted(list(set(data)))

    # 创建模型
    rnn = CharRNN(chars, hidden_size=100, seq_length=25, learning_rate=0.01)

    # 训练模型
    rnn.train(data, iterations=10000, print_interval=1000)

    # 生成文本
    seed_char = data[0]
    seed_idx = rnn.char_to_idx[seed_char]
    generated_text = rnn.sample(np.zeros((rnn.hidden_size, 1)), seed_idx, 500)
    print(f"Generated text:\n{generated_text}")


# 如果需要一个更复杂的LSTM版本实现，可以参考以下代码
class CharLSTM:
    def __init__(self, chars, hidden_size=100, seq_length=25, learning_rate=1e-1):
        """LSTM字符级语言模型实现"""
        self.chars = chars
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.learning_rate = learning_rate

        # 字符映射
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)

        # 初始化LSTM参数 (输入门, 遗忘门, 输出门, 候选记忆单元)
        # 为简洁起见，省略具体实现
        self.init_parameters()

    def init_parameters(self):
        """初始化LSTM参数"""
        # 输入权重
        self.Wf = np.random.randn(self.hidden_size, self.hidden_size + self.vocab_size) * 0.01
        self.Wi = np.random.randn(self.hidden_size, self.hidden_size + self.vocab_size) * 0.01
        self.Wo = np.random.randn(self.hidden_size, self.hidden_size + self.vocab_size) * 0.01
        self.Wc = np.random.randn(self.hidden_size, self.hidden_size + self.vocab_size) * 0.01

        # 偏置
        self.bf = np.zeros((self.hidden_size, 1))
        self.bi = np.zeros((self.hidden_size, 1))
        self.bo = np.zeros((self.hidden_size, 1))
        self.bc = np.zeros((self.hidden_size, 1))

        # 输出权重
        self.Why = np.random.randn(self.vocab_size, self.hidden_size) * 0.01
        self.by = np.zeros((self.vocab_size, 1))

        # 梯度缓存
        # 为简洁起见，省略Adagrad/Adam实现

    def forward_lstm_step(self, x, prev_h, prev_c):
        """单步LSTM前向传播"""
        # 连接输入和隐藏状态
        z = np.row_stack((prev_h, x))

        # 遗忘门
        f = self.sigmoid(np.dot(self.Wf, z) + self.bf)
        # 输入门
        i = self.sigmoid(np.dot(self.Wi, z) + self.bi)
        # 输出门
        o = self.sigmoid(np.dot(self.Wo, z) + self.bo)
        # 候选记忆单元
        c_candidate = np.tanh(np.dot(self.Wc, z) + self.bc)

        # 更新记忆单元
        next_c = f * prev_c + i * c_candidate
        # 更新隐藏状态
        next_h = o * np.tanh(next_c)

        # 输出
        y = np.dot(self.Why, next_h) + self.by
        p = np.exp(y) / np.sum(np.exp(y))

        return next_h, next_c, p

    def sigmoid(self, x):
        """Sigmoid激活函数"""
        return 1 / (1 + np.exp(-x))

    # 其余方法(backward, update_parameters, sample, train)省略
    # 实现思路与CharRNN类似，但需处理LSTM的特殊结构