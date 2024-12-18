import numpy as np
import torch
from collections import Counter
from sklearn import datasets
import torch.nn.functional as Fun
import matplotlib.pyplot as plt


# (1).数据集的读入
dataset = datasets.load_iris()  # 加载鸢尾花数据集
x_data = dataset.data  # 返回iris数据集所有输入特征
y_data = dataset.target  # 返回iris数据集所有标签
# print("x_data from datasets:", len(x_data))
print("y_data from datasets", y_data)

# (2).数据集乱序
np.random.seed(116)  # 使用相同的种子seed，使得乱序后的数据特征和标签仍然可以对齐
np.random.shuffle(x_data)  # 打乱数据集
np.random.seed(116)
np.random.shuffle(y_data)
# tf.random.set_seed(116)
print(y_data)
# (3).数据集不相交的训练集和测试集
x_train = x_data[:-30]  # 前120个数据作为训练集
y_train = y_data[:-30]  # 前120个标签作为训练集标签
x_test = x_data[-30:]  # 后30个数据集作为测试集
y_test = y_data[-30:]  # 后30个标签作为测试集标签

train_input = torch.FloatTensor(x_train)
test_input = torch.FloatTensor(x_test)
train_label = torch.LongTensor(y_train)
test_label = torch.LongTensor(y_test)
epochs = 500
loss_results = []
test_acc = []
# 2. 定义BP神经网络
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden1,n_hidden2, n_output):
        super(Net, self).__init__()
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden1)   # 定义
        self.hidden2 = torch.nn.Linear(n_hidden1, n_hidden2)  # 定义
        self.out = torch.nn.Linear(n_hidden2, n_output)  # 定义输出层网络
    def forward(self, x):  # 前馈规则
        # x = Fun.relu(self.hidden1(x))
        x = self.hidden1(x)
        x = Fun.relu(self.hidden2(x))  # 隐藏层的激活函数,采用relu,
        # 也可以采用sigmod, tanh
        x = self.out(x)  # 输出层不用激活函数
        return x


# 3. 定义优化器和损失函数
net = Net(n_feature=4, n_hidden1=20, n_hidden2=12, n_output=3)    # n_feature:输入的特征维度,n_hiddenb:神经元个数,n_output:输出的类别个数
print(net)
optimizer = torch.optim.SGD(net.parameters(), lr=0.02)  # 优化器选用随机梯度下降方式
loss_func = torch.nn.CrossEntropyLoss()  # 对于多分类一般采用的交叉熵损失函数,

def train():
    net.train()
    out1 = net(train_input)                 # 输入input,输出out
    loss = loss_func(out1, train_label)     # 输出与label对比
    print("epoch:{} loss:{}".format(epoch, loss))
    loss_results.append(loss.detach().numpy())  #取出loss数值
    optimizer.zero_grad()   # 梯度清零
    loss.backward()         # 前馈操作
    optimizer.step()        # 使用梯度优化器


def test():
    net.eval()
    test_loss = 0
    correct = 0
    n_classes = 3
    target_num = torch.zeros((1, n_classes))  # n_classes为分类任务类别数量
    predict_num = torch.zeros((1, n_classes))
    acc_num = torch.zeros((1, n_classes))
    with torch.no_grad():
        for step,(data, target) in enumerate(zip(test_input,test_label)):
            output = net(test_input)
            test_loss += Fun.nll_loss(output, test_label,
                                    size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            pre_mask = torch.zeros(output.size()).scatter_(1, pred.cpu().view(-1, 1), 1.)
            predict_num += pre_mask.sum(0)  # 得到数据中每类的预测量
            tar_mask = torch.zeros(output.size()).scatter_(1, target.data.cpu().view(-1, 1), 1.)
            target_num += tar_mask.sum(0)  # 得到数据中每类的数量
            acc_mask = pre_mask * tar_mask
            acc_num += acc_mask.sum(0)  # 得到各类别分类正确的样本数量

            # print(pred)
            correct += pred.eq(test_label.data.view_as(pred)).sum()
    recall = acc_num / target_num
    precision = acc_num / predict_num
    F1 = 2 * recall * precision / (recall + precision)
    accuracy = 100. * acc_num.sum(1) / target_num.sum(1)
    x = ['1', '2', '3']
    # plt.plot(x,recall.numpy().T,label='recall')
    # plt.plot(x,precision.numpy().T,label='precision')
    # plt.plot(x,F1.numpy().T,label='F1')
    # plt.legend()
    # plt.show()
    print('Test recall {}, precision {}, F1-score {}'.format(recall, precision, F1))
    # (2).得出结果
    out2 = net(test_input)  # out是一个计算矩阵，可以用Fun.softmax(out)转    化为概率矩阵
    prediction = torch.max(out2, 1)[1]  # 返回index  0返回原值
    pred_y = prediction.data.numpy()
    target_y = test_label.data.numpy()

    # (3). 衡量准确率
    accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
    test_acc.append(accuracy)
    print("莺尾花预测准确率：{}".format(accuracy))

for epoch in range(epochs):
    train()
    test()

# 绘制loss曲线
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.plot(loss_results, label="$Loss$")  # 逐点画出loss值并连线
plt.legend()
plt.savefig("./loss")
plt.show()

#  绘制Accuracy曲线
plt.title("Acc Curve")
plt.xlabel("Epoch")
plt.ylabel("Acc")
plt.plot(test_acc, label="$Accuracy$")  # 逐点画出test_acc值并连线
plt.legend()
plt.savefig("./acc")
plt.show()