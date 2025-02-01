import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from model.model import Xception, ShallowXception
import pandas as pd
from load_dataset import train_data_load
import os
import argparse
from visualization import visualization_with_tsne
import time

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'

def train_process(train_load, val_load, model, epoch_num, learning_rate, model_name, previous_epoch=0):
    # 检查cuda可用性
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'{device} is available')
    print("training model [{}]".format(model_name))

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)

    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []

    # t-SNE 可视化参数
    val_feature = []
    val_labels = []
    # t-SNE 可视化参数
    print("================================START TRAINING================================")
    for epoch in range(epoch_num):
        best_acc = 0.0
        best_model_weight = model.state_dict()
        train_loss = 0
        train_correct = 0
        train_times = 0
        val_loss = 0
        val_correct = 0
        val_times = 0
        for data, label in train_load:
            data, label = data.to(device), label.to(device)
            model.train()
            output = model.forward(data)
            y_hat = torch.argmax(output, dim=1)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(data)
            train_correct += torch.sum(label == y_hat)
            train_times += len(data)
        for data, label in val_load:
            data, label = data.to(device), label.to(device)
            model.eval()
            with torch.no_grad():
                output = model.forward(data)
                y_hat = torch.argmax(output, dim=1)
                loss = criterion(output, label)
                val_loss += loss.item() * len(data)
                val_correct += torch.sum(label == y_hat)
                val_times += len(data)

                # t-SNE 特征
                features = model.forward(data).cpu().numpy()
                val_feature.append(features)
                val_labels.append(label.cpu().numpy())
                # t-SNE 特征

        train_loss_history.append(train_loss / train_times)
        val_loss_history.append(val_loss / val_times)
        train_acc_history.append((train_correct / train_times).cpu().item())
        val_acc_history.append((val_correct / val_times).cpu().item())
        print("epoch {}, train loss:{:.4f}, train acc:{:.4f}".format(epoch, train_loss_history[-1], train_acc_history[-1]))
        print("epoch {}, val loss:{:.4f}, val acc:{:.4f}".format(epoch, val_loss_history[-1], val_acc_history[-1]))

        if val_acc_history[-1] * 100.0 > best_acc * 100.0:
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

    # t-SNE 特征合并
    val_feature = np.concatenate(val_feature, axis=0)
    val_labels = np.concatenate(val_labels, axis=0)
    # t-SNE 特征合并
    return result, val_feature, val_labels

def plt_acc_loss(result, modelname):
    # 利用时间戳保存文件名
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    unique_filename = f"{modelname}_{timestamp}_acc_loss.png"

    # 绘图
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(result["epoch"], result.train_loss_history, 'ro-', label="train loss")
    plt.plot(result["epoch"], result.val_loss_history, 'bs-', label="val loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.subplot(1, 2, 2)
    plt.plot(result["epoch"], result.train_acc_history, 'ro-', label="train acc")
    plt.plot(result["epoch"], result.val_acc_history, 'bs-', label="val acc")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.grid(True)
    plt.savefig(f"./figures/{unique_filename}")
    plt.show()
    print(f"Saved plot as {unique_filename}")

if __name__ == '__main__':
    model_name = ""

    # 参数解析
    parser = argparse.ArgumentParser(description="manual to this script")
    parser.add_argument("--c", type=bool, default=False, help="whether to train model by pretrained weight")
    parser.add_argument("--epoch", type=int, default=10, help="number of epochs")
    parser.add_argument("--model", type=str, default="Xception", help="which model(Xception or ShallowXception) to use")
    parser.add_argument("--bs", type=int, default=8, help="batch size, default=8")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate, default=1e-4")
    parser.add_argument("--v", type=bool, default=False, help="whether to show t-SNE visualization")
    args = parser.parse_args()
    batch_size = args.bs

    # 按照参数初始化模型
    if args.model == "Xception":
        model = Xception()
        model_name = model.name()
    elif args.model == "ShallowXception":
        model = ShallowXception()
        model_name = model.name()

    log_file = f"./log/{model_name}_logs.csv"
    #选择继续训练则读取checkpoint
    previous_epoch = 0  # 先前训练的epoch数
    if args.c:
        # 读取模型权重
        model.load_state_dict(torch.load('./weight/{}.pth'.format(model_name)))

        # 检查并读取日志文件
        if os.path.exists(log_file):
            previous_result = pd.read_csv(log_file)
            previous_epoch = len(previous_result["epoch"])
            print(f"Loaded existing log file: ./log/{log_file}")
        else:
            print(f"No existing log file: ./log/{log_file}")
            previous_result = None
        print("Continue Training")
    else:
        previous_result = None

    #获取参数epoch，lr
    num_epochs = args.epoch
    lr = args.lr

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
        visualization_with_tsne(val_features, val_labels, model_name)







