import torch
from model.model import Xception, ShallowXception
from load_dataset import test_data_load

def model_test_process(model, test_load):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    test_correct = 0
    test_times = 0
    model.eval()
    with torch.no_grad():
        for images, labels in test_load:
            images,labels = images.to(device),labels.to(device)

            labels = torch.argmax(labels,dim=1)

            output = model(images)
            y_hat = torch.argmax(output, dim=1)
            test_correct += torch.sum(y_hat == labels).item()
            test_times += images.size(0)
    print("测试了{}次，正确率为{:.4f}%".format(test_times, (test_correct / test_times)*100))

if __name__ == "__main__":
    model = ShallowXception(num_classes=6)
    model.load_state_dict(torch.load("./weight/ShallowXception.pth", weights_only = True))
    test_load = test_data_load(bs=8, classes=6)
    model_test_process(model, test_load)