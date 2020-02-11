import os
from tqdm import tqdm

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.nn import DataParallel

from datasets import URISC
from models import CASENet, ResNetUNet, DDS, DFF
from utils import Options, F_score, LR_Scheduler, focal_loss, fscore_loss, near_edge_loss


class Trainer:
    def __init__(self, args):
        self.args = args

        if "simple" in str.lower(args.dataset):
            self.dataset = "simple"
        else:
            assert "complex" in str.lower(args.dataset)
            self.dataset = "complex"

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

        trainset = URISC(path=args.dataset, mode="train", transform=self.transform,
                         crop_size=args.crop_size, augmentation=args.augmentation)
        valset = URISC(path=args.dataset, mode="val", transform=self.transform)
        self.trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                      drop_last=True, pin_memory=True, num_workers=16)
        self.valloader = DataLoader(valset, batch_size=1, shuffle=False)

        if args.model == "CASENet":
            self.model = DataParallel(CASENet(backbone=args.backbone)).cuda()
        elif args.model == "ResNetUNet":
            self.model = DataParallel(ResNetUNet(backbone=args.backbone)).cuda()
        elif args.model == "DDS":
            self.model = DataParallel(DDS(backbone=args.backbone)).cuda()
        elif args.model == "DFF":
            self.model = DataParallel(DFF(backbone=args.backbone)).cuda()
        else:
            # TODO add more models
            pass

        params_list = [{"params": self.model.module.backbone.parameters(), "lr": args.lr}]
        for name, param in self.model.module.named_parameters():
            if "backbone" not in name:
                params_list.append({"params": param, "lr": args.lr * args.lr_times})
        self.optimizer = Adam(params_list)
        self.lr_scheduler = LR_Scheduler(base_lr=args.lr, epochs=args.epochs,
                                         iters_each_epoch=len(self.trainloader), lr_times=args.lr_times)
        self.iterations = 0
        self.previous_best = 0.0

    def training(self, epoch):
        self.model.train()
        total_loss = 0.0
        tbar = tqdm(self.trainloader)
        for i, (image, target) in enumerate(tbar):
            self.lr_scheduler(self.optimizer, self.iterations, epoch)
            self.iterations += 1
            image, target = image.cuda(), target.cuda()
            predicted = self.model(image)
            self.optimizer.zero_grad()
            loss = 0.5 * fscore_loss(predicted, target) + \
                   0.5 * focal_loss(predicted, target, alpha=self.args.alpha, power=2) + \
                   self.args.edge_weight * near_edge_loss(predicted, target, kernel_size=self.args.kernel_size)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            tbar.set_description("Train loss: %.4f" % (total_loss / (i + 1)))

    def validation(self):
        self.model.eval()
        # evaluate on training set
        if args.eval_train:
            trainset = URISC(path=args.dataset, mode="train", transform=self.transform)
            trainloader = DataLoader(trainset, batch_size=1, shuffle=False)
            self.eval(trainloader)

        # evaluate on validation set
        mean_f_score = self.eval(self.valloader)

        if mean_f_score > self.previous_best:
            if self.previous_best != 0:
                os.remove(os.path.join(args.output, self.dataset, "models", "best_%.5f.pth" % self.previous_best))
            self.previous_best = mean_f_score
            torch.save(self.model.state_dict(), os.path.join(args.output, self.dataset,
                                                             "models", "best_%.5f.pth" % self.previous_best))

    def eval(self, dataloader):
        total_f_score, total_precison, total_recall = 0.0, 0.0, 0.0
        tbar = tqdm(dataloader, desc="\r")
        for t, (image, target) in enumerate(tbar):
            image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                predicted = self.model.module.tta_eval(image)
                predicted = (predicted > 0.5).float()
                f_score, precision, recall = F_score(predicted, target)
                total_f_score += f_score
                total_precison += precision
                total_recall += recall
                tbar.set_description("F-score: %.3f, Precision: %.3f, Recall: %.3f" %
                                     (total_f_score / (t + 1), total_precison / (t + 1), total_recall / (t + 1)))
        mean_f_score = total_f_score / len(dataloader)
        return mean_f_score


if __name__ == "__main__":
    args = Options().parse()
    trainer = Trainer(args)
    print("Total Epoches: %i" % (args.epochs))
    for epoch in range(args.epochs):
        print("\n=>Epoches %i, learning rate = %.4f, \t\t\t\t previous best = %.4f"
              % (epoch, trainer.optimizer.param_groups[0]["lr"], trainer.previous_best))
        trainer.training(epoch)
        trainer.validation()
