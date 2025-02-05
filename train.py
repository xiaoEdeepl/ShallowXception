# python官方模块
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import os
import time

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

# 自定义模块
from argument import parse_arguments
from model.model import Xception, ShallowXception
from load_dataset import train_data_load
from visualization import visualization_with_tsne, plt_acc_loss

RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
RESET = "\033[0m"
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'

def train_process(train_load, val_load, model, epoch_num, learning_rate, model_name, wd, previous_epoch=0):
    # 检查cuda可用性
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'{device} is available')
    print("training model [{}]".format(model_name))

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=wd)
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)
    num_classes = model.classes()

    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []
    val_auc_history = []

    # t-SNE 可视化参数
    val_features = []
    val_labels = []
    # t-SNE 可视化参数
    best_acc = 0.0
    # best_model_weight = model.state_dict()

    print("================================START TRAINING================================")

    for epoch in range(epoch_num):
        start_time = time.time()

        train_loss = 0
        train_correct = 0
        train_times = 0
        val_loss = 0
        val_correct = 0
        val_times = 0
        # 训练阶段
        for data, label in train_load:
            data, label = data.to(device), label.to(device)
            # 将 one-hot 标签转换为类别索引
            target = torch.argmax(label, dim=1)
            model.train()
            output = model(data)
            y_hat = torch.argmax(output, dim=1)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * data.size(0)
            train_correct += torch.sum(target == y_hat).item()
            train_times += data.size(0)

        # 验证阶段
        all_labels = []
        all_probs = []
        model.eval()
        with torch.no_grad():
            for data, label in val_load:
                data, label = data.to(device), label.to(device)
                target = torch.argmax(label, dim=1)

                output = model(data)
                y_hat = torch.argmax(output, dim=1)
                probs = torch.softmax(output, dim=1)
                loss = criterion(output, target)

                val_loss += loss.item() * data.size(0)
                val_correct += torch.sum(target == y_hat).item()
                val_times += data.size(0)

                all_labels.append(label.cpu().numpy())
                all_probs.append(probs.cpu().numpy())

            all_probs = np.concatenate(all_probs, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)
            all_labels_onehot = label_binarize(all_labels, classes=list(range(num_classes)))

            auc = roc_auc_score(all_labels_onehot, all_probs, multi_class='ovr')
            val_auc_history.append(auc)

        epoch_time = time.time() - start_time
        train_loss_history.append(round(train_loss / train_times, ndigits=4))
        val_loss_history.append(round(val_loss / val_times, ndigits=4))
        train_acc_history.append(round(train_correct / train_times, ndigits=4))
        val_acc_history.append(round(val_correct / val_times, ndigits=4))

        print(
            f"Epoch {epoch:03d} ,Time :{epoch_time:.1f}s | "
            f"Train Loss: {RED}{train_loss_history[-1]:.4f}{RESET} | "
            f"Train Acc: {GREEN}{train_acc_history[-1]:.4f}{RESET} | "
            f"Val Loss: {YELLOW}{val_loss_history[-1]:.4f}{RESET} | "
            f"Val Acc: {BLUE}{val_acc_history[-1]:.4f}{RESET} | "
            f"Val AUC: {GREEN}{val_auc_history[-1]:.4f}{RESET}"
        )

        # 如果在验证集上Acc更高，则更新模型
        if val_acc_history[-1] > best_acc:
            best_acc = val_acc_history[-1]
            best_model_weight = model.state_dict()
            torch.save(best_model_weight, './weight/{}.pth'.format(model_name))
            print("model updated with accuracy {:.4f}".format(best_acc))

    result = pd.DataFrame(data={
        "epoch":range(epoch_num),
        "train_loss_history":train_loss_history,
        "val_loss_history":val_loss_history,
        "train_acc_history":train_acc_history,
        "val_acc_history":val_acc_history,
    })

    result["epoch"] += previous_epoch

    # 使用最佳模型 t-SNE可视化
    model.load_state_dict(torch.load(f'./weight/{model_name}.pth'))
    model.eval()

    with torch.no_grad():
        for data, label in val_load:
            data,label = data.to(device), label.to(device)
            # 提取中间层特征
            features = model.extract_features(data)
            features = features.view(features.size(0), -1).cpu().numpy()
            val_features.append(features)
            val_labels.append(label.cpu().numpy())

    val_features = np.concatenate(val_features, axis=0)
    val_labels = np.concatenate(val_labels, axis=0)
    val_labels = np.argmax(val_labels, axis=1)

    return result, val_features, val_labels

if __name__ == '__main__':

    # 获取参数
    args = parse_arguments()
    batch_size = args.bs
    classes = args.classes
    wd = args.wd
    num_epochs = args.epoch
    lr = args.lr

    model_name = ""
    # 按照参数初始化模型
    if args.model == "Xception":
        model = Xception()
        model_name = model.name()
    elif args.model == "ShallowXception":
        model = ShallowXception(num_classes=classes)
        model_name = model.name()

    log_file = f"./log/{model_name}_logs.csv"
    #选择继续训练则读取checkpoint
    previous_epoch = 0  # 先前已训练的epoch数
    if args.c:
        # 读取模型权重
        model.load_state_dict(torch.load('./weight/{}.pth'.format(model_name)))
        # 检查并读取日志文件
        if os.path.exists(log_file):
            previous_result = pd.read_csv(log_file)
            previous_epoch = len(previous_result["epoch"])
            print(f"成功加载log文件: {log_file}，模型已训练 {previous_epoch} 个Epoch")
        else:
            print(f"No existing log file: {log_file}")
            previous_result = None
        print("Continue Training")
    else:
        previous_result = None



    # 读取数据集
    train_load, val_load = train_data_load(batch_size)
    # 训练
    result, val_features, val_labels = train_process(
        train_load,
        val_load,
        model,
        epoch_num=num_epochs,
        learning_rate=lr,
        model_name=model_name,
        wd=wd,
        previous_epoch = previous_epoch
    )

    # 更新已有的csv或保存新的csv文件
    if previous_result is not None:
        combined_result = pd.concat([previous_result, result], ignore_index=True)
        combined_result.to_csv(f"./log/{model_name}_logs.csv", index=False)
        print(f"Updated log file: ./log/{log_file}")
        plt_acc_loss(combined_result, model_name)
    else:
        result.to_csv(f"./log/{model_name}_logs.csv", index=False)
        print(f"Saved log file: ./log/{log_file}")
        plt_acc_loss(result, model_name)

    # t-SNE可视化
    if args.v:
        visualization_with_tsne(feature=val_features,
                                labels=val_labels,
                                model_name=model_name,
                                # label_dict={'df':0, 'f2f':1, 'fshift':2, 'fswap':3, 'nt':4, 'real':5}
                                label_dict={'real':1, 'fake':0},)







