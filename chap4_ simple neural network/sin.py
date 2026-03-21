import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子，确保结果可复现
np.random.seed(42)

class ReLUNetwork:
    """
    两层ReLU神经网络
    结构：输入层 -> 隐藏层(ReLU) -> 输出层(线性)
    """
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        """
        初始化网络参数
        
        参数:
        input_size: 输入维度
        hidden_size: 隐藏层神经元数量
        output_size: 输出维度
        learning_rate: 学习率
        """
        # 使用Xavier初始化
        limit1 = np.sqrt(6.0 / (input_size + hidden_size))
        self.W1 = np.random.uniform(-limit1, limit1, (input_size, hidden_size))
        self.b1 = np.zeros((1, hidden_size))
        
        limit2 = np.sqrt(6.0 / (hidden_size + output_size))
        self.W2 = np.random.uniform(-limit2, limit2, (hidden_size, output_size))
        self.b2 = np.zeros((1, output_size))
        
        self.lr = learning_rate
        self.grad_clip = 1.0
        
    def relu(self, x):
        """ReLU激活函数"""
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """ReLU函数的导数"""
        return (x > 0).astype(float)
    
    def forward(self, X):
        """
        前向传播
        """
        # 第一层：线性变换
        self.z1 = np.dot(X, self.W1) + self.b1
        # 激活函数
        self.a1 = self.relu(self.z1)
        # 第二层：线性变换
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        
        return self.z2
    
    def backward(self, X, y, output):
        """
        反向传播，计算梯度并更新参数
        """
        batch_size = X.shape[0]
        
        # 输出层梯度 (均方误差损失)
        dloss = 2 * (output - y) / batch_size
        
        # 检查梯度是否有效
        if np.any(np.isnan(dloss)) or np.any(np.isinf(dloss)):
            return
        
        # 第二层梯度
        dW2 = np.dot(self.a1.T, dloss)
        db2 = np.sum(dloss, axis=0, keepdims=True)
        
        # 反向传播到隐藏层
        dhidden = np.dot(dloss, self.W2.T)
        # ReLU激活函数的梯度
        relu_grad = self.relu_derivative(self.z1)
        dhidden = dhidden * relu_grad
        
        # 第一层梯度
        dW1 = np.dot(X.T, dhidden)
        db1 = np.sum(dhidden, axis=0, keepdims=True)
        
        # 梯度裁剪
        for grad in [dW1, db1, dW2, db2]:
            grad_norm = np.linalg.norm(grad)
            if grad_norm > self.grad_clip:
                grad *= self.grad_clip / (grad_norm + 1e-8)
        
        # 更新参数
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
    
    def train(self, X, y, epochs, batch_size, verbose=True):
        """
        训练网络
        """
        num_samples = X.shape[0]
        losses = []
        
        # 数据标准化
        self.X_mean = np.mean(X)
        self.X_std = np.std(X) + 1e-8
        X_normalized = (X - self.X_mean) / self.X_std
        
        for epoch in range(epochs):
            # 随机打乱数据
            indices = np.random.permutation(num_samples)
            X_shuffled = X_normalized[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0
            num_batches = 0
            
            # 小批量梯度下降
            for i in range(0, num_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # 前向传播
                output = self.forward(X_batch)
                
                # 检查输出是否有效
                if np.any(np.isnan(output)) or np.any(np.isinf(output)):
                    continue
                
                # 计算损失
                loss = np.mean((output - y_batch) ** 2)
                
                if np.isnan(loss) or np.isinf(loss):
                    continue
                    
                epoch_loss += loss
                num_batches += 1
                
                # 反向传播
                self.backward(X_batch, y_batch, output)
            
            if num_batches > 0:
                avg_loss = epoch_loss / num_batches
                losses.append(avg_loss)
                
                if verbose and (epoch + 1) % 500 == 0:
                    print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}')
        
        return losses
    
    def predict(self, X):
        """预测"""
        X_normalized = (X - self.X_mean) / self.X_std
        return self.forward(X_normalized)


def generate_data(num_points=1000, noise=0.05):
    """
    生成训练和测试数据
    """
    # 在 [-1, 1] 范围内均匀采样
    X = np.linspace(-1, 1, num_points).reshape(-1, 1)

    # 目标函数：y = np.sin(2πx) + 0.3 * sin(6πx)
    y = np.sin(2 * np.pi * X) + 0.3 * np.sin(6 * np.pi * X)
    # 添加少量噪声
    if noise > 0:
        y += np.random.randn(*y.shape) * noise
    
    return X, y


def plot_results(X_train, y_train, X_test, y_test, y_pred, losses):
    """
    绘制结果图
    """
    # 设置字体和负号显示
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
    plt.figure(figsize=(15, 5))
    
    # 子图1: 拟合效果
    plt.subplot(1, 2, 1)
    plt.scatter(X_train, y_train, alpha=0.3, s=10, label='Training Data', color='blue')
    plt.plot(X_test, y_test, 'g-', linewidth=2, label='True Function', alpha=0.7)
    plt.plot(X_test, y_pred, 'r--', linewidth=2, label='Predicted Function')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('ReLU Network Fitting Result')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图2: 损失曲线
    plt.subplot(1, 2, 2)
    # 过滤掉无效的损失值
    valid_losses = [l for l in losses if not (np.isnan(l) or np.isinf(l))]
    if valid_losses:
        plt.plot(valid_losses, 'b-', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.grid(True, alpha=0.3)
        
        # 使用对数坐标
        plt.yscale('log')
        
        # 获取当前坐标轴
        ax = plt.gca()
        
        # 设置y轴刻度格式，使其显示为科学计数法
        # 使用ScalarFormatter并设置科学计数法
        from matplotlib.ticker import ScalarFormatter, LogFormatter
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.get_major_formatter().set_scientific(True)
        ax.yaxis.get_major_formatter().set_powerlimits((-2, 3))
    
    plt.tight_layout()
    plt.show()


def evaluate_model(model, X_test, y_test):
    """
    评估模型性能
    """
    y_pred = model.predict(X_test)
    
    # 检查预测值是否有效
    if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
        print("Warning: Predictions contain invalid values")
        return float('nan'), float('nan'), y_pred
    
    mse = np.mean((y_pred - y_test) ** 2)
    mae = np.mean(np.abs(y_pred - y_test))
    
    print(f'\nModel Evaluation Results:')
    print(f'Mean Squared Error (MSE): {mse:.6f}')
    print(f'Mean Absolute Error (MAE): {mae:.6f}')
    
    return mse, mae, y_pred


def main():
    """
    主函数：数据生成、模型训练和评估
    """
    print("="*50)
    print("Two-layer ReLU Network Fitting sin Function")
    print("="*50)
    
    # 1. 生成数据
    print("\n1. Generating training and testing data...")
    X_train, y_train = generate_data(num_points=500, noise=0.03)
    X_test, y_test = generate_data(num_points=200, noise=0.0)
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Testing set size: {X_test.shape[0]}")
    print(f"Target function: f(x) = sin(x) + 0.1*sin(3x)")
    
    # 2. 创建模型
    print("\n2. Creating neural network model...")
    input_size = 1
    hidden_size = 32
    output_size = 1
    learning_rate = 0.01
    
    model = ReLUNetwork(input_size, hidden_size, output_size, learning_rate)
    print(f"Network structure: {input_size} -> {hidden_size} (ReLU) -> {output_size}")
    print(f"Learning rate: {learning_rate}")
    
    # 3. 训练模型
    print("\n3. Training...")
    epochs = 3000
    batch_size = 32
    
    losses = model.train(X_train, y_train, epochs, batch_size, verbose=True)
    
    # 4. 评估模型
    print("\n4. Evaluating model performance...")
    mse, mae, y_pred = evaluate_model(model, X_test, y_test)
    
    # 5. 可视化结果
    print("\n5. Plotting results...")
    plot_results(X_train, y_train, X_test, y_test, y_pred, losses)
    
    # 6. 分析隐藏层
    print("\n6. Analyzing hidden layer...")
    # 使用测试数据的前100个点分析隐藏层
    X_sample = X_test[:100]
    model.predict(X_sample)  # 这会更新隐藏层激活
    hidden_activations = model.a1
    
    # 安全检查
    if not np.any(np.isnan(hidden_activations)):
        activation_rate = np.mean(hidden_activations > 0)
        print(f"Hidden layer average activation rate: {activation_rate:.2%}")
        print(f"Hidden layer output mean: {np.mean(hidden_activations):.4f}")
        print(f"Hidden layer output std: {np.std(hidden_activations):.4f}")
    else:
        print("Hidden layer output contains invalid values")
    
    # 7. 计算拟合误差的统计信息
    print("\n7. Fitting error analysis...")
    errors = y_pred - y_test
    print(f"Error mean: {np.mean(errors):.6f}")
    print(f"Error std: {np.std(errors):.6f}")
    print(f"Error range: [{np.min(errors):.6f}, {np.max(errors):.6f}]")
    
    print("\n" + "="*50)
    print("Training completed!")
    print("="*50)
    

if __name__ == "__main__":
    main()