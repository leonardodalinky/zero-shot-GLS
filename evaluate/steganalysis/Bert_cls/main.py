import argparse
import os
import os.path as osp

import accelerate
import torch
from Bert_cls import BertCls
from DataLoader import BertDataset
from torch.utils.data import DataLoader
from utils import test, train

TMP_SAVES_DIR = f"{osp.dirname(__file__)}/../../../tmp_saves"

os.environ["HF_HOME"] = f"{TMP_SAVES_DIR}/hg_cache"


def get_args():
    parser = argparse.ArgumentParser(description="BerT")

    parser.add_argument(
        "-epochs", type=int, default=10, help="number of epochs for train [default:20]"
    )
    parser.add_argument("-save-dir", type=str, help="where to save the snapshot")
    parser.add_argument("-save-ckp", default=None, type=str, help="where to save the snapshot")
    parser.add_argument(
        "-load-dir", type=str, default=None, help="where to loading the trained model"
    )
    parser.add_argument(
        "-gen-path",
        type=str,
        help="The path of generated imdb data.",
    )
    parser.add_argument(
        "-gt-path",
        type=str,
        default="zero-shot-GLS/datasets/imdb/imdb.csv",
        help="The path of imdb data.",
    )
    # device
    parser.add_argument(
        "-no-cuda", action="store_true", default=False, help="disable the gpu [default:False]"
    )
    parser.add_argument(
        "-device", type=str, default="cuda", help="device to use for trianing [default:cuda]"
    )

    args = parser.parse_args()
    args.cuda = (not args.no_cuda) and torch.cuda.is_available()
    return args


def main():
    accelerate.utils.set_seed(2024)

    # Load args
    args = get_args()
    # Load Data
    train_data, test_data = BertDataset.load_data(args)
    train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=32, shuffle=True)

    # Load model
    # model: BertModel = AutoModel.from_pretrained("prajjwal1/bert-medium")
    # model.encoder.layer[0:4].requires_grad_(False)
    model = BertCls()

    if args.cuda:
        torch.device(args.device)
        model = model.cuda()

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-4)

    # Train
    print("\n--------{}--------".format(args.save_dir.split("/")[-1]))
    acc = []
    for epoch in range(args.epochs):
        print("\n--------training epochs: {}-----------".format(epoch))
        train(model, train_dataloader, optimizer)

        # Test
        print("--------testing-----------")
        acc.append(test(model, test_dataloader))
        print("Accuracy: {}".format(acc[-1]))

    # Save to file
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    with open(os.path.join(args.save_dir, "acc.txt"), "w") as f:
        f.write("Accuracy list: {}\n".format(acc))
        f.write("Best accuracy: {}".format(max(acc)))


if __name__ == "__main__":
    main()
