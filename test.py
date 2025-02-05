import torch
from model.model import Xception, ShallowXception
from load_dataset import Test_data_load
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
import numpy as np

def model_test_process(model, test_load, num_classes=6):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    test_correct = 0
    test_times = 0

    all_probs = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for images, labels in test_load:
            images,labels = images.to(device),labels.to(device)

            true_labels = torch.argmax(labels,dim=1)

            output = model(images)
            y_hat = torch.argmax(output, dim=1)

            test_correct += torch.sum(y_hat == true_labels).item()
            test_times += images.size(0)

            probs = torch.softmax(output, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    accuracy = test_correct / test_times

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    all_labels_onehot = label_binarize(all_labels, classes=list(range(num_classes)))

    auc = roc_auc_score(all_labels_onehot, all_probs, multi_class='ovr')
    print("测试了{}次，正确率为{:.4f}%，AUC分数为{:.4f}".format(test_times, accuracy*100, auc*100))

if __name__ == "__main__":
    model = ShallowXception(num_classes=6)
    model.load_state_dict(torch.load("./weight/ShallowXception.pth", weights_only = True))
    test_load = Test_data_load(bs=8, classes=6)
    model_test_process(model, test_load, num_classes=6)