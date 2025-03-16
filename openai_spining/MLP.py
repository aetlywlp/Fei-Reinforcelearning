import numpy as np


class NeuralNetwork:
    def __init__(self, layer_dims, learning_rate=0.1, lambda_reg=0):
        """
        初始化神经网络

        参数:
        layer_dims -- 包含各层维度的列表，例如[2,3,1]表示2个输入单元，3个隐藏单元，1个输出单元
        learning_rate -- 学习率
        lambda_reg -- 正则化参数
        """
        self.layer_dims = layer_dims
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg
        self.parameters = self.initialize_parameters()
        self.nl = len(layer_dims)  # 层数（包括输入层）

    def initialize_parameters(self):
        """
        随机初始化参数 (对称破缺)
        """
        np.random.seed(42)
        parameters = {}
        L = len(self.layer_dims)

        for l in range(1, L):
            # 小随机值初始化
            parameters['W' + str(l)] = np.random.randn(self.layer_dims[l], self.layer_dims[l - 1]) * 0.01
            parameters['b' + str(l)] = np.zeros((self.layer_dims[l], 1))

        return parameters

    def sigmoid(self, Z):
        """实现sigmoid激活函数"""
        return 1 / (1 + np.exp(-Z))

    def sigmoid_derivative(self, Z):
        """sigmoid函数的导数: f'(z) = f(z)(1-f(z))"""
        s = self.sigmoid(Z)
        return s * (1 - s)

    def forward_propagation(self, X):
        """
        前向传播计算

        参数:
        X -- 输入数据，形状为(输入特征数, 样本数)

        返回:
        AL -- 最后一层的激活值
        caches -- 包含每一层的Z值和激活值，用于反向传播
        """
        caches = []
        A = X
        L = self.nl - 1  # 层数（不包括输入层）

        # 存储输入层的激活值
        caches.append({"A": X})

        # 对每一层执行前向传播
        for l in range(1, L + 1):
            A_prev = A
            W = self.parameters['W' + str(l)]
            b = self.parameters['b' + str(l)]

            # 线性计算：Z = WA + b
            Z = np.dot(W, A_prev) + b

            # 非线性激活：A = sigmoid(Z)
            A = self.sigmoid(Z)

            # 存储当前层的信息用于反向传播
            cache = {'Z': Z, 'A': A}
            caches.append(cache)

        return A, caches

    def compute_cost(self, AL, Y):
        """
        计算成本函数 (带L2正则化)
        """
        m = Y.shape[1]  # 样本数

        # 计算交叉熵成本
        logprobs = np.multiply(np.log(AL), Y) + np.multiply(np.log(1 - AL), 1 - Y)
        cost_without_reg = -np.sum(logprobs) / m

        # 添加L2正则化项
        L = self.nl - 1
        reg_cost = 0
        for l in range(1, L + 1):
            reg_cost += np.sum(np.square(self.parameters['W' + str(l)]))
        reg_cost = self.lambda_reg / (2 * m) * reg_cost

        cost = cost_without_reg + reg_cost

        return cost

    def backward_propagation(self, AL, Y, caches):
        """
        实现反向传播算法

        参数:
        AL -- 输出层的激活值
        Y -- 真实标签
        caches -- 前向传播保存的缓存

        返回:
        grads -- 包含梯度的字典
        """
        grads = {}
        L = self.nl - 1  # 层数（不包括输入层）
        m = AL.shape[1]  # 样本数

        # 初始化反向传播
        # 1. 计算输出层的误差 (步骤1)
        # δ^(nl) = -(y-a^(nl)) ⊙ f'(z^(nl))
        dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

        # 当前层(输出层)的误差
        current_cache = caches[L]
        Z = current_cache['Z']
        dZ = dAL * self.sigmoid_derivative(Z)

        # 获取前一层的激活值
        A_prev = caches[L - 1]['A']
        W = self.parameters['W' + str(L)]

        # 计算当前层的梯度 (步骤3)
        # dW^(l) = δ^(l+1) ⋅ (a^(l))^T / m + λW^(l)/m
        grads['dW' + str(L)] = np.dot(dZ, A_prev.T) / m + (self.lambda_reg / m) * W
        # db^(l) = δ^(l+1) / m
        grads['db' + str(L)] = np.sum(dZ, axis=1, keepdims=True) / m

        # 2. 从L-1到1反向循环计算梯度 (步骤2)
        for l in reversed(range(1, L)):
            # 获取当前层的缓存
            current_cache = caches[l]
            Z = current_cache['Z']

            # 获取前一层的激活值
            A_prev = caches[l - 1]['A']
            W_next = self.parameters['W' + str(l + 1)]

            # 计算当前层的误差 δ^(l) = ((W^(l))^T ⋅ δ^(l+1)) ⊙ f'(z^(l))
            dZ_next = dZ  # 保存下一层的dZ
            dA = np.dot(W_next.T, dZ_next)
            dZ = dA * self.sigmoid_derivative(Z)

            # 当前层的权重
            W = self.parameters['W' + str(l)]

            # 计算当前层的梯度 (步骤3)
            grads['dW' + str(l)] = np.dot(dZ, A_prev.T) / m + (self.lambda_reg / m) * W
            grads['db' + str(l)] = np.sum(dZ, axis=1, keepdims=True) / m

        return grads

    def update_parameters(self, grads):
        """使用梯度下降更新参数"""
        L = self.nl - 1

        # 更新每一层的参数
        for l in range(1, L + 1):
            self.parameters['W' + str(l)] -= self.learning_rate * grads['dW' + str(l)]
            self.parameters['b' + str(l)] -= self.learning_rate * grads['db' + str(l)]

    def train(self, X, Y, num_iterations=3000, print_cost=True):
        """
        训练神经网络
        """
        costs = []

        # 梯度下降训练
        for i in range(num_iterations):
            # 前向传播
            AL, caches = self.forward_propagation(X)

            # 计算成本
            cost = self.compute_cost(AL, Y)

            # 反向传播计算梯度
            grads = self.backward_propagation(AL, Y, caches)

            # 更新参数
            self.update_parameters(grads)

            # 记录成本
            if print_cost and i % 100 == 0:
                costs.append(cost)
                print(f"迭代 {i}, 成本: {cost:.6f}")

        return self.parameters, costs

    def predict(self, X):
        """使用训练好的模型进行预测"""
        A, _ = self.forward_propagation(X)
        predictions = (A > 0.5).astype(int)
        return predictions


# 示例：用神经网络解决XOR问题
def xor_example():
    # 生成XOR数据
    X = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])  # 2个特征，4个样本
    Y = np.array([[0, 1, 1, 0]])  # XOR输出

    # 创建神经网络 [2, 4, 1] - 2个输入，4个隐藏单元，1个输出
    nn = NeuralNetwork([2, 4, 1], learning_rate=0.5, lambda_reg=0.01)

    # 训练网络
    parameters, costs = nn.train(X, Y, num_iterations=5000)

    # 预测并计算准确率
    predictions = nn.predict(X)
    print("\n预测结果:")
    print(predictions)
    print(f"准确率: {np.mean(predictions == Y) * 100:.2f}%")

    return nn, costs


# 运行XOR示例
if __name__ == "__main__":
    print("训练神经网络解决XOR问题...\n")
    nn, costs = xor_example()