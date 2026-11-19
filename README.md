# 实验四：基于多层感知机（MLP）的 MNIST 手写数字识别

#### 202410315058-姚钧耀
文件说明

本实验的代码在 `mlp_mnist.py` 文件中，实验所需数据集 MNIST 会在运行时自动下载到 `data` 文件夹。

## 一、实验目的

基于多层感知机（MLP，即全连接神经网络）实现 MNIST 手写数字数据集的分类识别。掌握 PyTorch 中全连接层（Linear）的构建、ReLU 激活函数的使用，理解张量展平（Flatten）操作的意义，以及模型训练与评估（Train/Eval 模式切换）的标准流程。

## 二、实验内容

### 1. 导入必要库

- `torch`、`torch.nn`、`torch.nn.functional`、`torch.optim`：PyTorch 核心及神经网络、优化器模块。
- `numpy`、`matplotlib.pyplot`：数据处理与可视化。
- `torchvision`：提供 MNIST 数据集及数据转换工具。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
2. 超参数设置与数据加载
将批大小（Batch Size）设置为 2048，并自动检测是否可用 GPU（cuda）。使用 DataLoader 加载 MNIST 训练集和测试集，并通过 ToTensor() 将图像转为张量。

python
batch_size = 2048
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                   ])),
    batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
    ])),
    batch_size=batch_size, shuffle=True)
3. 多层感知机（MLP）网络定义
本实验构建一个包含一个隐藏层的简单 MLP：

输入层：784（对应 28×28 的图像展平后维度）

隐藏层：128 个神经元，使用 ReLU 激活函数

输出层：10（对应 0-9 十个数字类别）

python
class mlp(nn.Module):
    def __init__(self):
        super(mlp, self).__init__()
        self.l1 = nn.Linear(784, 128)
        self.l2 = nn.Linear(128, 10)

    def forward(self, x):
        a1 = self.l1(x)
        x1 = F.relu(a1)
        a2 = self.l2(x1)
        return a2

model = mlp().to(device)
# 使用随机梯度下降(SGD)优化器，学习率设为0.01
optimizer = optim.SGD(model.parameters(), lr=0.01)
4. 模型训练与测试
训练阶段：

调用 model.train() 开启训练模式。

关键操作：由于 MLP 只能接收一维向量，必须使用 x.view(x.shape[0], -1) 将 (batch_size, 1, 28, 28) 的图像张量展平为 (batch_size, 784)。

使用交叉熵损失函数进行反向传播和参数更新。

测试阶段：

调用 model.eval() 开启评估模式。

使用 torch.no_grad() 关闭梯度计算以节省内存并加速。

统计测试损失并计算准确率。

python
epochs = 30
for epoch in range(epochs):
    # --- 训练阶段 ---
    model.train()
    for batch_idx, (x, y) in enumerate(train_loader):
        # 展平图像张量: (batch_size, 1, 28, 28) -> (batch_size, 784)
        x, y = x.view(x.shape[0], -1).to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = F.cross_entropy(output, y)
        loss.backward()
        optimizer.step()

    # --- 测试阶段 ---
    model.eval()
    correct = 0
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            x, y = x.view(x.shape[0], -1).to(device), y.to(device)
            output = model(x)
            test_loss += F.cross_entropy(output, y).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(y.view_as(pred)).sum().item()

    test_loss = test_loss / (batch_idx + 1)
    acc = correct / len(test_loader.dataset)
    print('epoch:{}, loss:{:.4f}, acc:{:.4f}'.format(epoch, test_loss, acc))
```
## 三、实验结果与分析

### 1. 训练与测试性能分析

- **模型收敛**：实验使用 SGD 优化器（学习率 0.01），在 30 个 epoch 下，模型能够有效收敛。通常在前几个 epoch 准确率就能迅速攀升至 90% 以上，最终稳定在 97% - 98% 左右。
- **Batch Size 的影响**：由于 `batch_size` 设为了较大的 2048，意味着每个 epoch 只有约 30 次参数更新（60000/2048）。虽然单次更新梯度稳定，但整体更新次数较少，因此收敛过程相对平缓。
- **泛化能力**：MLP 模型没有空间特征提取能力，全连接层参数量巨大，但由于 MNIST 任务较为简单，模型未出现严重过拟合。

### 2. 模型结构分析

- **展平操作的必然性**：与 CNN（如 LeNet-5）不同，MLP 不具备处理二维空间结构的能力。`x.view(x.shape[0], -1)` 将每张图像的空间像素打散成一维向量，全连接层通过密集的权重连接进行特征映射。
- **激活函数的作用**：隐藏层使用 `F.relu()`，相比 Sigmoid，ReLU 能够有效缓解梯度消失问题，加速了模型在 784 维高维特征上的训练过程。

### 3. 优缺点分析

- **优点**：结构简单、易于实现、推理速度极快（仅涉及矩阵乘法）。
- **缺点**：丢失了图像的二维空间位置信息（相邻像素关系），导致特征提取能力远弱于卷积神经网络，面对复杂图像任务时性能会大幅下降。

## 四、实验总结与心得

### 1. 技术收获

- **掌握了 PyTorch 的模型生命周期**：深刻理解了 `model.train()` 与 `model.eval()` 的切换意义。虽然在简单的 MLP 中两者的表现差异不大，但这是处理 Dropout 和 BatchNorm 层（如后续实验中的 CNN）时的标准规范。
- **理解了张量形状（Shape）的变换**：明白了全连接层仅接受 `(Batch, Features)` 形状的输入，掌握了使用 `view()` 或 `reshape()` 进行维度展平的方法。
- **熟悉了优化器差异**：相较于实验五中收敛极快的 Adam 优化器，本实验使用的 SGD 更加基础，让我直观感受到了不同优化器在收敛速度上的差异。

### 2. 实践体会

- **代码细节**：原参考代码在展平操作时存在一定的排版混乱，补全时需仔细确认 `x.view(x.shape[0], -1)` 的维度参数，否则会导致矩阵乘法维度不匹配报错。
- **从全连接到卷积的思维转变**：通过对比实验四（MLP）与实验五（LeNet-5），我直观地认识到：MLP 把图像“拍扁”处理，而 CNN 通过卷积核保留了局部感受野，这使得 CNN 在图像识别任务中天生具备优势。这为后续学习更深的卷积网络打下了认知基础。
