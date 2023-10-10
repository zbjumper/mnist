import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision

import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64*7*7, 1024)  # 两个池化，所以是7*7而不是14*14
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)
#         self.dp = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(-1, 64 * 7 * 7)  # 将数据平整为一维的
        x = F.relu(self.fc1(x))
#         x = self.fc3(x)
#         self.dp(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
#         x = F.log_softmax(x,dim=1) NLLLoss()才需要，交叉熵不需要
        return x


# 批数量
BATCH_SIZE = 32
# 学习率
LR = 0.01
# 训练次数
EPOCH = 5
# 测试数量
CESHI = 2500

train_data = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True
)

test_data = torchvision.datasets.MNIST(
    root='./data',
    train=False
)

train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

test_x = torch.unsqueeze(test_data.data, dim=1).type(
    torch.FloatTensor)[:CESHI]/255
test_y = test_data.targets[:CESHI]
# print('current ago yum')
# print(train_data.data.size())
# print(test_data.data.size())

# net = nn.Sequential(nn.Flatten(), nn.Linear(784, 256),nn.ReLU(),
#                     nn.Linear(256, 64), nn.Linear(64, 10))
net = CNN()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = net.to(device)

test_x, test_y = test_x.to(device), test_y.to(device)

optimizer = torch.optim.SGD(net.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)

        # 计算输出
        output = net(x)
        # 计算损失
        loss = loss_func(output, y)

        # 重置梯度
        optimizer.zero_grad()

        # 反向传播， 使用链式反向求导的方法，一次计算模型中每个参数（即权重）的梯度
        loss.backward()
        optimizer.step()

        if step % 100 == 99:
            test_output = net(test_x)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = (pred_y == test_y).sum().item()/CESHI
            print('step:', step + 1, '| epoch:', epoch, '| loss:',
                  loss.data.cpu().numpy(), '| accuracy:', accuracy)

test_output = net(test_x)
print("test_output.shape:", test_output.shape)
pred_y = torch.max(test_output, 1)[1].data.squeeze()
print('真实:', test_y.cpu().numpy())
print('预测:', pred_y.cpu().numpy())
accuracy = (pred_y == test_y).sum().item()/CESHI
print('accuracy:', accuracy)
