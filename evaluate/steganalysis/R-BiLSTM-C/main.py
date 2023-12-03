import argparse
import datetime
import os
import sys

import DataLoader
import R_BI_C
import torch
import torch.nn as nn
import torchtext.data as data
import train
from torchtext.vocab import Vectors

parser = argparse.ArgumentParser(description="R_BI_C")

# learning
parser.add_argument(
    "-batch-size", type=int, default=64, help="batch size for training [default: 64]"
)
parser.add_argument("-lr", type=float, default=0.001, help="initial learning rate [default:0.001]")
parser.add_argument("-epochs", type=int, default=20, help="number of epochs for train [default:20]")
parser.add_argument(
    "-log-interval", type=int, default=1, help="how many epochs to wait defore logging train status"
)
parser.add_argument(
    "-test-interval",
    type=int,
    default=1,
    help="how many epochs to wait defore testing [default:100]",
)
parser.add_argument(
    "-save-interval",
    type=int,
    default=5,
    help="how many epochs to wait before saving [default:500]",
)
# parser.add_argument(
#     "-early-stop", type=int, default=100, help="iteration numbers to stop without performace boost"
# )
parser.add_argument(
    "-save-best", type=bool, default=True, help="whether to save when get best performance"
)
parser.add_argument("-save-dir", type=str, help="where to save the snapshot")
parser.add_argument("-save-ckp", default=None, type=str, help="where to save the snapshot")
parser.add_argument("-load-dir", type=str, default=None, help="where to loading the trained model")

# data
parser.add_argument(
    "-shuffle",
    action="store_true",
    default=False,
    help="shuffle the data every epoch [default:False]",
)
# parser.add_argument('-train-cover-dir', type=str, default='cover.txt',
# 					help='the path of train cover data. [default:cover.txt]')
# parser.add_argument('-train-stego-dir', type=str, default='1bpw.txt',
# 					help='the path of train stego data. [default:1bpw.txt]')
# parser.add_argument('-test-cover-dir', type=str, default='cover.txt',
# 					help='the path of tset cover data. [default:cover.txt]')
# parser.add_argument('-test-stego-dir', type=str, default='1bpw.txt',
# 					help='the path of test stego data. [default:1bpw.txt]')
parser.add_argument(
    "-gen-path",
    type=str,
    help="The path of generated imdb data.",
)
parser.add_argument(
    "-gt-path",
    type=str,
    default="/home/lyy/workspace/zero-shot-GLS/datasets/imdb/imdb.csv",
    help="The path of imdb data.",
)
# model
parser.add_argument(
    "-num-layers", type=int, default=3, help="the number of LSTM layers [default:3]"
)
parser.add_argument(
    "-kernel-sizes", type=str, default=[3, 5, 7], help="the sizes of kernels of CNN layers"
)
parser.add_argument(
    "-kernel-num", type=int, default=128, help="the number of each CNN kernels [default:100]"
)
parser.add_argument(
    "-embed-dim", type=int, default=300, help="number of embedding dimension [defualt:128]"
)
parser.add_argument(
    "-hidden-size", type=int, default=300, help="the number of hidden unit [defualt:300]"
)
parser.add_argument(
    "-LSTM_dropout", type=float, default=0.5, help="the probability for SLTM dropout [defualt:0.5]"
)
parser.add_argument(
    "-CNN_dropout", type=float, default=0.5, help="the probability for CNN dropout [defualt:0.5]"
)


# device
parser.add_argument(
    "-no-cuda", action="store_true", default=False, help="disable the gpu [default:False]"
)
parser.add_argument(
    "-device", type=str, default="cuda", help="device to use for trianing [default:cuda]"
)
parser.add_argument(
    "-idx-gpu", type=str, default="0", help="the number of gpu for training [default:0]"
)

# option
parser.add_argument(
    "-test", default=False, help="train or test [default:False]", action="store_true"
)

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.idx_gpu


def data_loader(text_field, label_field, args, **kwargs):
    train_data, valid_data = DataLoader.MyData.split(text_field, label_field, args, "train")

    vectors = Vectors(name="glove.6B.300d.txt", cache="glove_weight")
    text_field.build_vocab(train_data, valid_data, vectors=vectors)

    label_field.build_vocab(train_data, valid_data)
    train_iter, valid_iter = data.Iterator.splits(
        (train_data, valid_data), batch_sizes=(args.batch_size, len(valid_data)), **kwargs
    )

    return train_iter, valid_iter


# load data
print("\nLoading data...")
text_field = data.Field(lower=True, fix_length=40)
label_field = data.Field(sequential=False)
train_iter, valid_iter = data_loader(text_field, label_field, args, device=args.device, sort=False)

if args.test:
    test_data = DataLoader.MyData.split(text_field, label_field, args, "test")
    test_iter = data.Iterator.splits(
        [test_data], batch_sizes=[len(test_data)], device=args.device, sort=False
    )[0]


# update args and print
args.embed_num = len(text_field.vocab)
args.class_num = len(label_field.vocab) - 1
args.cuda = (not args.no_cuda) and torch.cuda.is_available()
del args.no_cuda

# print('\nParameters: ')
# for attr, value in sorted(args.__dict__.items()):
# 	print('\t{}={}'.format(attr.upper(), value))


# model
model = R_BI_C.R_BI_C(args, text_field)

# initializing model
for name, w in model.named_parameters():
    if "embed" not in name:
        if "fc1.weight" in name:
            nn.init.xavier_normal_(w)

        elif "weight" in name and "conv" in name:
            nn.init.normal_(w, 0.0, 0.1)

        if "bias" in name:
            nn.init.constant_(w, 0)

if args.load_dir is not None:
    print("\nLoading model from {}...".format(args.load_dir))
    model.load_state_dict(torch.load(args.load_dir))

if args.cuda:
    torch.device(args.device)
    model = model.cuda()

## Caculate the number of parameters of the loaded model
# total_params = sum(p.numel() for p in model.parameters())
# print('Model_size: ', total_params)
# sys.exit()


# training phase
if not args.test:
    print("\n----------training------------")
    train.train(train_iter, valid_iter, model, args)


# testing phase
else:
    print("\n----------testing------------")
    print("Loading test model from {}...".format(args.save_ckp))
    models = []
    # files = sorted(os.listdir(args.save_ckp))
    file = args.save_ckp
    model.load_state_dict(torch.load(file))
    # train.data_eval(test_iter, model, args)
    acc, r, p, f = train.data_eval(test_iter, model, args)
    # for name in files:
    #     if name.endswith(".pt"):
    #         models.append(name)
    # model_steps = sorted([int(m.split("_")[-1].split(".")[0]) for m in models])
    # ACC, R, P, F1 = 0, 0, 0, 0
    # for step in model_steps[-3:]:
    # best_model = "best_steps_{}.pt".format(step)
    # m_path = os.path.join(args.save_dir, best_model)
    # print("the {} model is loaded...".format(m_path))
    # model.load_state_dict(torch.load(m_path))
    # acc, r, p, f = train.data_eval(test_iter, model, args)
    # train.data_eval(test_iter, model, args)
    ACC = acc
    R = r
    P = p
    F1 = f

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    with open(os.path.join(args.save_dir, "result.txt"), "a") as f:
        f.write("The average testing accuracy: {:.4f} \n".format(ACC))
        f.write("The average testing recall: {:.4f} \n".format(R))
        f.write("The average testing precious: {:.4f} \n".format(P))
        f.write("The average testing F1_sorce: {:.4f} \n".format(F1))
