import torch
from model.model import biXception, ShallowXception
from load_dataset import test_data_load

def model_test_process(model, test_load):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    test_correct = 0
    test_times = 0

    with torch.no_grad():
        for images, labels in test_load:
            images = images.to(device)
            labels = labels.to(device)
            model.eval()

            output = model.forward(images)
            y_hat = torch.argmax(output, dim=1)
            test_correct += torch.sum(y_hat == labels)
            test_times += len(images)
    print("测试了{}次，正确率为{:.4f}%".format(test_times, (test_correct / test_times)*100))

if __name__ == "__main__":
    model = ShallowXception()
    model.load_state_dict(torch.load("./weight/ShallowXception.pth", weights_only = False))
    test_load = test_data_load(8)
    model_test_process(model, test_load)