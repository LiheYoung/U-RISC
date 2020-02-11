import argparse
import os


class Options:
    def __init__(self):
        parser = argparse.ArgumentParser("U-RISC COMPETITION")
        parser.add_argument("--dataset", type=str, default="simple",
                            help="training dataset: simple(default) or complex")
        parser.add_argument("--model", type=str, default="CASENet",
                            help="network model type (default: CASENet)")
        parser.add_argument("--batch-size", type=int, default=1,
                            help="batch size for training (default: 1)")
        parser.add_argument("--epochs", type=int, default=200,
                            help="training epochs (default: 200)")
        parser.add_argument("--backbone", type=str, default="resnet50",
                            help="network backbone (default: resnet50)")
        parser.add_argument("--lr", type=float, default=0.0014,
                            help="learning rate (default: 0.0014)")
        parser.add_argument("--alpha", type=float, default=0.70,
                            help="alpha of focal loss (default: 0.70)")
        parser.add_argument("--lr-times", type=float, default=10.0,
                            help="learning rate of head over backbone (default: 10)")
        parser.add_argument("--aug", dest="augmentation", action="store_true",
                            help="augment training samples")
        parser.add_argument("--crop-size", type=int, default=960,
                            help="crop size of original image (default: 960)")
        parser.add_argument("--eval-train", dest="eval_train", action="store_true",
                            help="evaluating model on training set")
        parser.add_argument("--split-size", type=int, default=4000,
                            help="split size of large testing image in complex dataset")
        parser.add_argument("--k", type=int, default=1,
                            help="k-th fold of cross validation")
        parser.add_argument("--kernel-size", type=int, default=9)
        parser.add_argument("--edge-weight", type=float, default=0)
        parser.add_argument("--output", type=str, default="outdir")

        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        if args.dataset == "simple":
            path = "data/datasets/simple/val"
            filenames = os.listdir(path)
            for filename in filenames:
                os.rename(os.path.join(path, filename), os.path.join(path.replace("val", "train"), filename))
            filenames = os.listdir(path.replace("val", "train"))
            filenames.sort()
            os.rename(os.path.join(path.replace("val", "train"), filenames[-args.k]),
                      os.path.join(path, filenames[-args.k]))
        args.dataset = os.path.join("data/datasets", str.lower(args.dataset))
        return args
