import argparse

def parse_arguments():
    # 参数解析
    parser = argparse.ArgumentParser(description="manual to this script")

    # 布尔参数需要使用 `store_true`
    parser.add_argument("--c", action="store_true", help="whether to train model by pretrained weight")
    parser.add_argument("--v", action="store_true", help="whether to show t-SNE visualization")

    # 其他参数
    parser.add_argument("--epoch", type=int, default=10, help="number of epochs")
    parser.add_argument("--model", type=str, default="xception", choices=["xception", "shallowxception"], help="which model to use")
    parser.add_argument("--bs", type=int, default=8, help="batch size, default=8")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate, default=1e-4")
    parser.add_argument("--classes", type=int, default=2, help="number of classes, default=2")
    parser.add_argument("--wd", type=float, default=1e-5, help="weight decay, default=1e-5")

    # 数据集参数
    parser.add_argument("--dataset", type=str, default="ff", choices=["ff", "dfdc", "cdf"], help="train dataset (ff, dfdc, cdf)")

    args = parser.parse_args()
    return args

# 运行解析
if __name__ == "__main__":
    args = parse_arguments()
    print(args)
