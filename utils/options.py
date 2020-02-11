import argparse
import os


class Options:
    def __init__(self):
        parser = argparse.ArgumentParser("U-RISC COMPETITION")
        parser.add_argument("--dataset", type=str, default="simple",
                            help="training dataset: simple(default) or complex")
        parser.add_argument("--model", type=str)
        parser.add_argument("--batch-size", type=int, default=4,
                            help="batch size for training (default: 4)")
        parser.add_argument("--epochs", type=int)
        parser.add_argument("--backbone", type=str)
        parser.add_argument("--lr", type=float, default=0.0014,
                            help="learning rate (default: 0.0014)")
        parser.add_argument("--alpha", type=float, default=0.70,
                            help="alpha of focal loss (default: 0.70)")
        parser.add_argument("--lr-times", type=float, default=10.0,
                            help="learning rate of head over backbone (default: 10)")
        parser.add_argument("--aug", dest="augmentation", action="store_true",
                            help="augment training samples")
        parser.add_argument("--crop-size", type=int)
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
            if args.epochs is None:
                args.epochs = 200
            if args.backbone is None:
                args.backbone = "resnet50"
            if args.crop_size is None:
                args.crop_size = 960
            if args.model is None:
                args.model = "DFF"
        elif args.dataset == "complex":
            if args.epochs is None:
                args.epochs = 45
            if args.backbone is None:
                args.backbone = "resnet152"
            if args.crop_size is None:
                args.crop_size = 1280
            if args.model is None:
                args.model = "CASENet"

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
