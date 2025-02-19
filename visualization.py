# T-SNE Visualization
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import time
from model.model import shallowxception, xception

import torch
from load_dataset import train_data_load
import os

os.environ["LOKY_MAX_CPU_COUNT"] = "16"

def plt_acc_loss(result, modelname:str):
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

def visualization_with_tsne(feature, labels, model_name, label_dict, save=True):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"./figures/{model_name}_tsne_{timestamp}.png"

    tsne = TSNE(n_components=2, random_state=42, n_jobs=-1)
    # 执行降维
    reduce_features = tsne.fit_transform(feature)

    # 创建画布
    plt.figure(figsize=(12, 8))
    unique_labels = np.unique(labels)
    cmap = plt.cm.get_cmap("tab10")
    colors = [cmap(i/len(unique_labels)) for i in range(len(unique_labels))]

    # 创建反向字典（数值到类别名称）
    reverse_dict = {v: k for k, v in label_dict.items()} if label_dict else None

    for i, label in enumerate(unique_labels):
        idx = labels == label
        # 根据 label_dict 动态获取类名
        class_name = reverse_dict.get(label, str(label)) if reverse_dict else f"Class {label}"

        plt.scatter(
            reduce_features[idx, 0],
            reduce_features[idx, 1],
            color=colors[i],
            label=class_name,  # 直接显示类名
            alpha=0.6,
            edgecolors="w",
            linewidths=0.5
        )

    plt.legend(
        title="Classes",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        borderaxespad=0.,
    )
    plt.title(f"t-SNE Visualization - {model_name}\n({timestamp})")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    if save:
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"可视化结果已保存至：{filename}")
    plt.show()
    plt.close()

if __name__ == "__main__":
    model = shallowxception()
    model.load_state_dict(torch.load("weight/shallowxception.pth"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    _, val_load = train_data_load(bs=16)

    val_features = []
    val_labels = []
    model.eval()
    with torch.no_grad():
        for data, label in val_load:
            data, label = data.to(device), label.to(device)
            features = model.extract_features(data)
            features = features.view(features.shape[0], -1).cpu().numpy()
            val_features.append(features)
            val_labels.append(label.cpu().numpy())

    val_features = np.concatenate(val_features, axis=0)
    val_labels = np.concatenate(val_labels, axis=0)
    val_labels = np.argmax(val_labels, axis=1)

    # 调用可视化函数
    visualization_with_tsne(
        feature=val_features,
        labels=val_labels,
        model_name="shallowxception",
        save=False,
        label_dict={'fake': 0, 'real': 1}
    )