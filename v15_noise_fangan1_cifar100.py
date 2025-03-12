# ============================================================================
# SplitfedV1 (SFLV1) learning: ResNet18 on HAM10000
# HAM10000 dataset: Tschandl, P.: The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions (2018), doi:10.7910/DVN/DBW86T

# We have three versions of our implementations
# Version1: without using socket and no DP+PixelDP
# Version2: with using socket but no DP+PixelDP
# Version3: without using socket but with DP+PixelDP

# This program is Version1: Single program simulation
# ============================================================================
import torch
from torch import nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from scipy.spatial.distance import cdist
import numpy as np
from sklearn.mixture import GaussianMixture
import math
import os.path
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
from glob import glob
from pandas import DataFrame
from torch.utils.data import Subset
import random
import numpy as np
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    print(torch.cuda.get_device_name(0))

# ===================================================================
program = "SFLV1 ResNet18 on HAM10000"
print(f"---------{program}----------")  # this is to identify the program in the slurm outputs files

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(0)


# To print in color -------test/train of the client side
def prRed(skk): print("\033[91m {}\033[00m".format(skk))


def prGreen(skk): print("\033[92m {}\033[00m".format(skk))


# ===================================================================
# No. of users
num_users = 50
epochs = 5
epochs2 = 195
epochs3 = 200
frac = 1  # participation of clients; if 1 then 100% clients participate in SFLV1
lr = 0.01


# def gmm_divide_entropy(A):
# # 将数据转换为二维数组
#     X = np.array(A).reshape(-1, 1)
#
#     # 创建 GMM 模型，设置为 2 个组件
#     gmm = GaussianMixture(n_components=2)
#
#     # 拟合 GMM 模型
#     gmm.fit(X)
#
#     # 获取每个数据点的标签（0 或 1，表示所属的类别）
#     labels = gmm.predict(X)
#
#     # 获取每个类别的均值
#     means = gmm.means_.flatten()
#
#     # 判断哪个均值较大，较大的均值为类别 1，较小的均值为类别 0
#     if means[0] > means[1]:
#         labels = 1 - labels  #
#     return labels,gmm.predict_proba(X)


def gmm_divide_lid(A):
    # 将数据转换为二维数组
    X = np.array(A).reshape(-1, 1)

    # 创建 GMM 模型，设置为 2 个组件
    gmm = GaussianMixture(n_components=2)

    # 拟合 GMM 模型
    gmm.fit(X)

    # 获取每个数据点的标签（0 或 1，表示所属的类别）
    labels = gmm.predict(X)

    # 获取每个类别的均值
    means = gmm.means_.flatten()

    if means[0] <= means[1]:
        labels = 1 - labels  #
    return labels, gmm.predict_proba(X)


def linear_map(data, new_min=0, new_max=10):
    # 获取数据中的最小值和最大值
    old_min = min(data)
    old_max = max(data)

    # 使用线性映射公式进行转换
    mapped_data = [(new_max - new_min) * (x - old_min) / (max(old_max - old_min,1e-10)) + new_min for x in data]

    return mapped_data


def calculate_entropy(user_labels, epsilon=1e-10):
    """
    计算标签熵，考虑零频率标签的数量
    :param label_frequencies: 每种标签的出现频率列表
    :param epsilon: 防止对数计算中的零值
    :return: 标签熵
    """
    b = []
    for i in range(len(user_labels)):
        b.append([])
    for j in range(100):
        for i in b:
            i.append(0)
    for i in range(len(user_labels)):
        for j in user_labels[i]:
            b[i][j] = b[i][j] + 1
    print(b)
    # label_sum=[]
    label_more=[]
    label_less=[]
    for i in b:
        # label_sum.append(sum(i))
        i.sort(reverse=True)
        n_length=len(i)
        label_more.append(sum(i[:n_length//2])/sum(i))
        label_less.append(sum(i[n_length // 2:])/sum(i))
    print(label_more)
    print(label_less)
    entropys = []
    label_more = np.array(label_more)
    label_less = np.array(label_less)
    for i in range(len(label_more)):
        if label_less[i]==0:
            print(111111)
            label_less[i]=1e-10
        # print(label_more[i])
        entropys.append(-label_more[i]*np.log(label_more[i])-label_less[i]*np.log(label_less[i]))
    return entropys


def lid_term(X, batch, k=20):
    eps = 1e-6
    X = np.asarray(X, dtype=np.float32)

    batch = np.asarray(batch, dtype=np.float32)
    f = lambda v: - k / (np.sum(np.log(v / (v[-1] + eps))) + eps)
    distances = cdist(X, batch)

    # get the closest k neighbours
    sort_indices = np.apply_along_axis(np.argsort, axis=1, arr=distances)[:, 1:k + 1]
    m, n = sort_indices.shape
    idx = np.ogrid[:m, :n]
    idx[1] = sort_indices
    # sorted matrix
    distances_ = distances[tuple(idx)]
    lids = np.apply_along_axis(f, axis=1, arr=distances_)
    return lids


def get_output(loader, net1, net2, latent=False, criterion=None):
    net1.eval()
    net2.eval()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            images = images.to(device)
            labels = labels.to(device)
            labels = labels.long()
            if latent == False:
                outputs = net1(images)
                outputs = net2(outputs)
                outputs = F.softmax(outputs, dim=1)
            else:
                outputs = net1(images, True)
                outputs = net2(outputs, True)
            loss = criterion(outputs, labels)
            if i == 0:
                output_whole = np.array(outputs.cpu())
                loss_whole = np.array(loss.cpu())
            else:
                output_whole = np.concatenate((output_whole, outputs.cpu()), axis=0)
                loss_whole = np.concatenate((loss_whole, loss.cpu()), axis=0)
    if criterion is not None:
        return output_whole, loss_whole
    else:
        return output_whole


# =====================================================================================================
#                           Client-side Model definition
# =====================================================================================================
# Model at client side
class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet18_client_side(nn.Module):
    def __init__(self, ResidualBlock, num_classes=100):
        super(ResNet18_client_side, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64, 3, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 4, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 6, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 3, stride=2)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # out = F.avg_pool2d(out, 4)
        # out = out.view(out.size(0), -1)
        # out = self.fc(out)
        return out


net_glob_client = ResNet18_client_side(ResidualBlock)
net_client_local = ResNet18_client_side(ResidualBlock)

net_glob_client.to(device)
print(net_glob_client)


# =====================================================================================================
#                           Server-side Model definition
# =====================================================================================================
# Model at server side


class ResNet18_server_side(nn.Module):
    def __init__(self, ResidualBlock, num_classes=100):
        super(ResNet18_server_side, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64, 3, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 4, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 6, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 3, stride=2)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        # out = self.conv1(x)
        # out = self.layer1(out)
        # out = self.layer2(x)
        # out = self.layer3(x)
        # out = self.layer4(out)
        out = F.avg_pool2d(x, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


net_glob_server = ResNet18_server_side(ResidualBlock)  # 7 is my numbr of classes
net_server_local = ResNet18_server_side(ResidualBlock)

net_glob_server.to(device)
print(net_glob_server)

# ===================================================================================
# For Server Side Loss and Accuracy
loss_train_collect = []
acc_train_collect = []
loss_test_collect = []
acc_test_collect = []
batch_acc_train = []
batch_loss_train = []
batch_acc_test = []
batch_loss_test = []
run_time = []

criterion = nn.CrossEntropyLoss()
count1 = 0
count2 = 0


# ====================================================================================================
#                                  Server Side Program
# ====================================================================================================
# Federated averaging: FedAvg
def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


def calculate_accuracy(fx, y):
    preds = fx.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = 100.00 * correct.float() / preds.shape[0]
    return acc


# to print train - test together in each round-- these are made global
acc_avg_all_user_train = 0
loss_avg_all_user_train = 0
loss_train_collect_user = []
acc_train_collect_user = []
loss_test_collect_user = []
acc_test_collect_user = []

w_glob_server = net_glob_server.state_dict()
w_locals_server = []
w_locals_biaoji = [0 for i in range(int(num_users / 50))]
# client idx collector
idx_collect = []
l_epoch_check = False
fed_check = False
# Initialization of net_model_server and net_server (server-side model)
net_model_server = [net_glob_server for i in range(int(num_users / 50))]
net_server = copy.deepcopy(net_model_server[0]).to(device)


# optimizer_server = torch.optim.Adam(net_server.parameters(), lr = lr)

# Server-side function associated with Training
def train_server(fx_client, y, l_epoch_count, l_epoch, idx, len_batch):
    global net_model_server, criterion, optimizer_server, device, batch_acc_train, batch_loss_train, l_epoch_check, fed_check
    global loss_train_collect, acc_train_collect, count1, acc_avg_all_user_train, loss_avg_all_user_train, idx_collect, w_locals_server, w_glob_server, net_server
    global loss_train_collect_user, acc_train_collect_user, lr, w_locals_biaoji, net_server_local
    net_server = copy.deepcopy(net_model_server[int(idx / 50)]).to(device)
    net_server.train()
    optimizer_server = torch.optim.SGD(net_server.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    # train and update
    optimizer_server.zero_grad()

    fx_client = fx_client.to(device)
    y = y.to(device)

    # ---------forward prop-------------
    fx_server = net_server(fx_client)

    # calculate loss
    loss = criterion(fx_server, y)
    # calculate accuracy
    acc = calculate_accuracy(fx_server, y)

    # --------backward prop--------------
    loss.backward()
    dfx_client = fx_client.grad.clone().detach()
    optimizer_server.step()

    batch_loss_train.append(loss.item())
    batch_acc_train.append(acc.item())

    # Update the server-side model for the current batch
    net_model_server[int(idx / 50)] = copy.deepcopy(net_server)
    net_server_local = copy.deepcopy(net_server)
    # count1: to track the completion of the local batch associated with one client
    count1 += 1
    if count1 == len_batch:
        acc_avg_train = sum(batch_acc_train) / len(batch_acc_train)  # it has accuracy for one batch
        loss_avg_train = sum(batch_loss_train) / len(batch_loss_train)

        batch_acc_train = []
        batch_loss_train = []
        count1 = 0

        prRed('Client{} Train => Local Epoch: {} \tAcc: {:.3f} \tLoss: {:.4f}'.format(idx, l_epoch_count, acc_avg_train,
                                                                                      loss_avg_train))

        # copy the last trained model in the batch
        w_server = net_server.state_dict()

        # If one local epoch is completed, after this a new client will come
        if l_epoch_count == l_epoch - 1:
            w_locals_biaoji[int(idx / 50)] += 1
            l_epoch_check = True  # to evaluate_server function - to check local epoch has completed or not
            # We store the state of the net_glob_server()
            if w_locals_biaoji[int(idx / 50)] == 50:
                w_locals_server.append(copy.deepcopy(w_server))

            # we store the last accuracy in the last batch of the epoch and it is not the average of all local epochs
            # this is because we work on the last trained model and its accuracy (not earlier cases)

            # print("accuracy = ", acc_avg_train)
            acc_avg_train_all = acc_avg_train
            loss_avg_train_all = loss_avg_train

            # accumulate accuracy and loss for each new user
            loss_train_collect_user.append(loss_avg_train_all)
            acc_train_collect_user.append(acc_avg_train_all)

            # collect the id of each new user
            if idx not in idx_collect:
                idx_collect.append(idx)
                # print(idx_collect)

        # This is for federation process--------------------
        if len(idx_collect) == num_users:
            fed_check = True  # to evaluate_server function  - to check fed check has hitted
            # Federation process at Server-Side------------------------- output print and update is done in evaluate_server()
            # for nicer display

            w_glob_server = FedAvg(w_locals_server)

            # server-side global model update and distribute that model to all clients ------------------------------
            net_glob_server.load_state_dict(w_glob_server)
            net_model_server = [net_glob_server for i in range(int(num_users / 50))]
            w_locals_biaoji = [0 for i in range(int(num_users / 50))]
            w_locals_server = []
            idx_collect = []

            acc_avg_all_user_train = sum(acc_train_collect_user) / len(acc_train_collect_user)
            loss_avg_all_user_train = sum(loss_train_collect_user) / len(loss_train_collect_user)

            loss_train_collect.append(loss_avg_all_user_train)
            acc_train_collect.append(acc_avg_all_user_train)

            acc_train_collect_user = []
            loss_train_collect_user = []

    # send gradients to the client
    return dfx_client


# Server-side functions associated with Testing
def evaluate_server(fx_client, y, idx, len_batch, ell):
    global net_model_server, criterion, batch_acc_test, batch_loss_test, check_fed, net_server, net_glob_server
    global loss_test_collect, acc_test_collect, count2, num_users, acc_avg_train_all, loss_avg_train_all, w_glob_server, l_epoch_check, fed_check
    global loss_test_collect_user, acc_test_collect_user, acc_avg_all_user_train, loss_avg_all_user_train

    net = copy.deepcopy(net_glob_server).to(device)
    # print(len(net_model_server))
    # net=net_model_server[int(idx/10)]
    net.eval()

    with torch.no_grad():
        fx_client = fx_client.to(device)
        y = y.to(device)
        # ---------forward prop-------------
        fx_server = net(fx_client)

        # calculate loss
        loss = criterion(fx_server, y)
        # calculate accuracy
        acc = calculate_accuracy(fx_server, y)

        batch_loss_test.append(loss.item())
        batch_acc_test.append(acc.item())

        count2 += 1
        if count2 == len_batch:
            acc_avg_test = sum(batch_acc_test) / len(batch_acc_test)
            loss_avg_test = sum(batch_loss_test) / len(batch_loss_test)

            batch_acc_test = []
            batch_loss_test = []
            count2 = 0

            prGreen('Client{} Test =>                   \tAcc: {:.3f} \tLoss: {:.4f}'.format(idx, acc_avg_test,
                                                                                             loss_avg_test))

            # if a local epoch is completed
            if l_epoch_check:
                l_epoch_check = False

                # Store the last accuracy and loss
                acc_avg_test_all = acc_avg_test
                loss_avg_test_all = loss_avg_test

                loss_test_collect_user.append(loss_avg_test_all)
                acc_test_collect_user.append(acc_avg_test_all)

            # if federation is happened----------
            if fed_check:
                fed_check = False
                print("------------------------------------------------")
                print("------ Federation process at Server-Side ------- ")
                print("------------------------------------------------")

                acc_avg_all_user = sum(acc_test_collect_user) / len(acc_test_collect_user)
                loss_avg_all_user = sum(loss_test_collect_user) / len(loss_test_collect_user)

                loss_test_collect.append(loss_avg_all_user)
                acc_test_collect.append(acc_avg_all_user)
                acc_test_collect_user = []
                loss_test_collect_user = []

                print("====================== SERVER V1==========================")
                print(' Train: Round {:3d}, Avg Accuracy {:.3f} | Avg Loss {:.3f}'.format(ell, acc_avg_all_user_train,
                                                                                          loss_avg_all_user_train))
                print(' Test: Round {:3d}, Avg Accuracy {:.3f} | Avg Loss {:.3f}'.format(ell, acc_avg_all_user,
                                                                                         loss_avg_all_user))
                print("==========================================================")

    return


# ==============================================================================================================
#                                       Clients-side Program
# ==============================================================================================================
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


# Client-side functions associated with Training and Testing
class Client(object):
    def __init__(self, net_client_model, idx, lr, device, dataset_train=None, dataset_test=None, idxs=None,
                 idxs_test=None):
        self.idx = idx
        self.device = device
        self.lr = lr
        self.local_ep = 5
        # self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset_train, idxs), batch_size=128, shuffle=True)
        self.ldr_test = DataLoader(DatasetSplit(dataset_test, idxs_test), batch_size=100, shuffle=True)

    def train(self, net):
        net.train()
        optimizer_client = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

        for iter in range(self.local_ep):
            len_batch = len(self.ldr_train)
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer_client.zero_grad()
                # ---------forward prop-------------
                fx = net(images)
                client_fx = fx.clone().detach().requires_grad_(True)

                # Sending activations to server and receiving gradients from server
                dfx = train_server(client_fx, labels, iter, self.local_ep, self.idx, len_batch)

                # --------backward prop -------------
                fx.backward(dfx)
                optimizer_client.step()

            # prRed('Client{} Train => Epoch: {}'.format(self.idx, ell))

        return net.state_dict()

    def evaluate(self, net, ell):
        net.eval()

        with torch.no_grad():
            len_batch = len(self.ldr_test)
            for batch_idx, (images, labels) in enumerate(self.ldr_test):
                images, labels = images.to(self.device), labels.to(self.device)
                # ---------forward prop-------------
                fx = net(images)

                # Sending activations to server
                evaluate_server(fx, labels, self.idx, len_batch, ell)

            # prRed('Client{} Test => Epoch: {}'.format(self.idx, ell))

        return
    # =====================================================================================================


# dataset_iid() will create a dictionary to collect the indices of the data samples randomly for each client
# IID HAM10000 datasets will be created based on this


def predict_softlabel(dataset,model_client, model_server):
    trainloader = DataLoader(dataset, batch_size=128, shuffle=False)
    model_client.eval()
    model_server.eval()
    all_preds = []
    with torch.no_grad():
        for data in trainloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # 获取模型输出
            outputs = model_client(inputs)
            outputs = model_server(outputs)

            # 获取最大值的索引作为预测标签
            _, predicted = torch.max(outputs, 1)

            # 将预测结果和真实标签逐一添加到列表中
            for i in range(len(predicted)):
                all_preds.append(predicted[i].item())  # 预测标签
    return all_preds


def cifar_user_dataset(dataset, num_users, noniid_fraction):
    """
    Sample a 'fraction' of non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :param fraction:
    :return:
    """

    # initialization
    total_items = len(dataset)
    num_noniid_items = len(dataset) * noniid_fraction
    num_iid_items = total_items - num_noniid_items
    dict_users = list()
    for ii in range(num_users):
        dict_users.append(list())
    idxs = [i for i in range(len(dataset))]

    # IID
    if num_iid_items != 0:
        per_user_iid_items = int(num_iid_items / num_users)
        for ii in range(num_users):
            tmp_set = set(np.random.choice(idxs, per_user_iid_items, replace=False))
            dict_users[ii] += tmp_set
            idxs = list(set(idxs) - tmp_set)

    # NON-IID
    if num_noniid_items != 0:

        num_shards = num_users  # each user has one shard
        per_shards_num_imgs = int(num_noniid_items / num_shards)
        idx_shard = [i for i in range(num_shards)]
        labels = list()
        for ii in range(len(idxs)):
            labels.append(dataset[idxs[ii]][1])
        print(labels)
        # sort labels
        idxs_labels = np.vstack((idxs, labels))
        idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
        # for i in range(len(idxs_labels)):
        #     print('aaaaaaaaaaaaaaaaaaaaaaaaaaa')
        #     print(idxs_labels[i])
        idxs = idxs_labels[0, :]

        # divide and assign
        i = 0
        while idx_shard:
            print(idx_shard)
            rand_idx = np.random.choice(idx_shard, 1, replace=False)
            rand_idx[0] = idx_shard[0]
            # rand_idx.append(idx_shard[0])
            print(rand_idx)
            idx_shard = list(set(idx_shard) - set(rand_idx))
            dict_users[i].extend(idxs[int(rand_idx) * per_shards_num_imgs: (int(rand_idx) + 1) * per_shards_num_imgs])
            i = divmod(i + 1, num_users)[1]

    '''
    for ii in range(num_users):
        tmp = list()
        for jj in range(len(dict_users[ii])):
            tmp.append(dataset[dict_users[ii][jj]][1])
        tmp.sort()
        print(tmp)
    '''
    return dict_users


def dataset_iid(dataset, num_users):
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


# =============================================================================
#                         Data loading
# =============================================================================

mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)

# 数据预处理和增强
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
dataset_train = datasets.CIFAR100('./data', train=True, download=True, transform=transform_train)
dataset_test = datasets.CIFAR100('./data', train=False, download=True, transform=transform_test)
predict_dataset = datasets.CIFAR100('./data', train=True, download=True, transform=transform_test)
# dataset_train = SkinData(train, transform = train_transforms)
# dataset_test = SkinData(test, transform = test_transforms)

# ----------------------------------------------------------------
# with open('beta=0.1.pkl', 'rb') as file:
#     dict_users=pickle.load(file)
# dict_users=cifar_user_dataset(dataset_train,num_users,0)
zhenshizaosheng=[]
with open('cifar100_n=50_alpha=i.txt', 'r') as file:
    content = file.read()
dict_users = eval(content)
dict_users_test = dataset_iid(dataset_test, num_users)
gailv = 0.6
noise_num = int(gailv * num_users)
clean_num = int(num_users - gailv * num_users)
gailvliebiao = [0] * noise_num + [1] * clean_num  # 0是噪声用户,1是干净用户
random.shuffle(gailvliebiao)
for i in range(num_users):
    # print(i)
    # print(len(dataset_train.targets))
    if gailvliebiao[i] == 1:
        continue
    print(i)
    for j in dict_users[i]:
        probability = 0.1
        pre_target = dataset_train.targets[j]
        if random.random() < probability:
            zhenshizaosheng.append(j)
            noisy_y = random.randint(0, 99)
            while noisy_y == pre_target:
                noisy_y = random.randint(0, 99)
            dataset_train.targets[j] = noisy_y

all_users_labels = []
for i in range(num_users):
    user_label = []
    for j in dict_users[i]:
        user_label.append(dataset_train.targets[j])
    all_users_labels.append(user_label)
label_entropy = calculate_entropy(all_users_labels)
print(label_entropy)
label_entropy = linear_map(label_entropy)
print(label_entropy)
# ------------ Training And Testing  -----------------
net_glob_client.train()
# copy weights
w_glob_client = net_glob_client.state_dict()
LID_client = np.zeros(num_users)
# b=None
loss_all_user={}
for iter in range(epochs):
    LID_client = np.zeros(num_users)
    m = max(int(frac * num_users), 1)
    idxs_users = np.random.choice(range(num_users), m, replace=False)
    w_locals_client = []

    for idx in idxs_users:
        local = Client(net_glob_client, idx, lr, device, dataset_train=dataset_train, dataset_test=dataset_test,
                       idxs=dict_users[idx], idxs_test=dict_users_test[idx])
        # Training ------------------
        w_client = local.train(net=copy.deepcopy(net_glob_client).to(device))
        w_locals_client.append(copy.deepcopy(w_client))

        # Testing -------------------
        local.evaluate(net=copy.deepcopy(net_glob_client).to(device), ell=iter)

        net_client_local.load_state_dict(copy.deepcopy(w_client))
        a, b = get_output(DataLoader(DatasetSplit(dataset_train, dict_users[idx]), batch_size=128, shuffle=False),
                          copy.deepcopy(net_client_local).to(device), copy.deepcopy(net_server_local).to(device), False,
                          nn.CrossEntropyLoss(reduction='none'))
        loss_all_user[idx]=b
        LID_local = list(lid_term(a, a))
        LID_client[idx] = np.mean(LID_local)
    w_glob_client = FedAvg(w_locals_client)
    print(LID_client)
    LID_client = linear_map(LID_client)
    print(LID_client)

    # for i in range(len(LID_client)):
    #     LID_client[i]=LID_client[i]+label_entropy[i]
    # print(LID_client)

    net_glob_client.load_state_dict(w_glob_client)

leibie_lid, gailv_lid = gmm_divide_lid(LID_client)
leibie_entropy, _ = gmm_divide_lid(label_entropy)
leibie_final = leibie_lid#这里先手动定义为leibie_lid，运行时手动切换


noise_clients=[i for i, value in enumerate(leibie_lid) if value == 0]
clean_clients=[i for i, value in enumerate(leibie_lid) if value == 1]

# noise_clients=[2, 3, 4, 5, 6, 8, 11, 13, 14, 15, 16, 18, 20, 21, 23, 24, 26, 27, 28, 30, 31, 32, 33, 34, 35, 40, 41, 42, 43, 46, 47, 48, 49, 50, 53, 54, 55, 56, 59, 61, 62, 63, 64, 69, 70, 73, 75, 77, 78, 82, 83, 85, 86, 87, 88, 89, 90, 93, 95, 96, 97, 98]
# clean_clients=[0, 1, 7, 9, 10, 12, 17, 19, 22, 25, 29, 36, 37, 38, 39, 44, 45, 51, 52, 57, 58, 60, 65, 66, 67, 68, 71, 72, 74, 76, 79, 80, 81, 84, 91, 92, 94, 99]
# nni=[i for i, value in enumerate(leibie_entropy) if value == 2]
print(noise_clients)
print(clean_clients)
# print(nni)
# noise_clients=[i for i, value in enumerate(leibie_lid) if value == 1]
# clean_clients=[i for i, value in enumerate(leibie_lid) if value == 0]
# print(noise_clients)
# print(clean_clients)
# relabel_clients=[]
relabel_dicts=[]
for client_id in noise_clients:
        gmm_loss = GaussianMixture(n_components=2, random_state=1).fit(np.array(loss_all_user[client_id]).reshape(-1, 1))
        labels_loss = gmm_loss.predict(np.array(loss_all_user[client_id]).reshape(-1, 1))
        gmm_clean_label_loss = np.argsort(gmm_loss.means_[:, 0])[0]

        pred_n = np.where(labels_loss.flatten() != gmm_clean_label_loss)[0]
        print(pred_n)
        sample_idx = np.array(list(dict_users[client_id]))
        # dataset_client = Subset(dataset_train, sample_idx)
        # loader_relabel = torch.utils.data.DataLoader(dataset=dataset_client, batch_size=100, shuffle=False)
        # print(type(sample_idx))
        loss = np.array(loss_all_user[client_id])
        # local_output, _ = get_output(loader_relabel, copy.deepcopy(net_glob_client).to(device), copy.deepcopy(net_glob_server).to(device), False,nn.CrossEntropyLoss(reduction='none'))
        relabel_idx = (-loss).argsort()[:int(len(pred_n)*0.3)]
        # relabel_idx = list(set(np.where(np.max(local_output, axis=1) > 0.5)[0])) #& set(relabel_idx))
        # print(relabel_idx)
        for i in relabel_idx:
            relabel_dicts.append(dict_users[client_id][i])
zhenshizaosheng.sort()
relabel_dicts.sort()
set1=set(zhenshizaosheng)
set2=set(relabel_dicts)
intersection=len(set1&set2)
union=len(set1|set2)
print(zhenshizaosheng)
print(relabel_dicts)
print(union/len(zhenshizaosheng))
print(union/len(relabel_dicts))
print(intersection/len(zhenshizaosheng))
print(intersection/len(relabel_dicts))
new_dict_users=[]
for i in range(num_users):
    new_dict_users.append([])
    for j in dict_users[i]:
        if j in relabel_dicts:
            pass
        else:
            new_dict_users[i].append(j)      
client_soft_label=None
server_soft_label=None 
client_soft_label_baoliu=None
server_soft_label_baoliu=None  
soft_acc=0
for iter in range(epochs2):
    m = max(int(frac * num_users), 1)
    idxs_users = np.random.choice(range(num_users), m, replace=False)
    w_locals_client = []

    for idx in idxs_users:
        # if leibie_final[idx] == 0:
        #     continue
        local = Client(net_glob_client, idx, lr, device, dataset_train=dataset_train, dataset_test=dataset_test,
                       idxs=new_dict_users[idx], idxs_test=dict_users_test[idx])
        # Training ------------------
        w_client = local.train(net=copy.deepcopy(net_glob_client).to(device))
        # if idx in clean_clients:
        w_locals_client.append(copy.deepcopy(w_client))

        # Testing -------------------
        local.evaluate(net=copy.deepcopy(net_glob_client).to(device), ell=iter)
    if soft_acc<acc_test_collect[-1]:
        soft_acc=acc_test_collect[-1]
        print(soft_acc)
        client_soft_label=copy.deepcopy(client_soft_label_baoliu)
        server_soft_label=copy.deepcopy(server_soft_label_baoliu)
    w_glob_client = FedAvg(w_locals_client)
    net_glob_client.load_state_dict(w_glob_client)
    client_soft_label_baoliu=copy.deepcopy(net_glob_client)
    server_soft_label_baoliu=copy.deepcopy(net_glob_server)
    
# net_glob_client= torch.load('cifar10_client_06_040.pt')
# net_glob_server=torch.load('cifar10_server_06_040.pt')

print(predict_softlabel(predict_dataset, copy.deepcopy(client_soft_label), copy.deepcopy(server_soft_label)))
soft_labels = predict_softlabel(predict_dataset, copy.deepcopy(client_soft_label), copy.deepcopy(server_soft_label))
for i in relabel_dicts:
    dataset_train.targets[i] = soft_labels[i]

for iter in range(epochs3):
    # if iter==100:
    #     torch.save(net_glob_client,'cifar10_client_03_040.pt')
    #     torch.save(net_glob_server,'cifar10_server_03_040.pt')
    m = max(int(frac * num_users), 1)
    idxs_users = np.random.choice(range(num_users), m, replace=False)
    w_locals_client = []

    for idx in idxs_users:
        local = Client(net_glob_client, idx, lr, device, dataset_train=dataset_train, dataset_test=dataset_test,
                       idxs=dict_users[idx], idxs_test=dict_users_test[idx])
        # Training ------------------
        w_client = local.train(net=copy.deepcopy(net_glob_client).to(device))
        w_locals_client.append(copy.deepcopy(w_client))

        # Testing -------------------
        local.evaluate(net=copy.deepcopy(net_glob_client).to(device), ell=iter)

    w_glob_client = FedAvg(w_locals_client)
    net_glob_client.load_state_dict(w_glob_client)
# ===================================================================================

print("Training and Evaluation completed!")

# ===============================================================================
# Save output data to .excel file (we use for comparision plots)
round_process = [i for i in range(1, len(acc_train_collect) + 1)]
print(loss_train_collect)
print(loss_test_collect)
print(acc_train_collect)
print(acc_test_collect)
# torch.save(net_glob_client,'cifar10_client_1_000.pt')
# torch.save(net_glob_server,'cifar10_server_1_000.pt')

# =============================================================================
#                         Program Completed
# =============================================================================