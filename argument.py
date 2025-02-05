import argparse

def parse_arguments():
    # 参数解析
    parser = argparse.ArgumentParser(description="manual to this script")
    parser.add_argument("--c", type=bool, default=False, help="whether to train model by pretrained weight")
    parser.add_argument("--epoch", type=int, default=10, help="number of epochs")
    parser.add_argument("--model", type=str, default="Xception", help="which model(Xception or ShallowXception) to use")
    parser.add_argument("--bs", type=int, default=8, help="batch size, default=8")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate, default=1e-4")
    parser.add_argument("--v", type=bool, default=False, help="whether to show t-SNE visualization")
    parser.add_argument("--classes", type=int, default=2, help="number of classes, default=2")
    parser.add_argument("--wd", type=float, default=1e-5, help="weight decay, default=1e-5")
    args = parser.parse_args()
    return args