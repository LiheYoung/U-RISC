class LR_Scheduler:
    # learning rate scheduler: lr = base_lr * (1 - iter/total_iter) ** power
    # power is set to be 0.9 by default
    def __init__(self, base_lr, epochs, iters_each_epoch, power=0.9, lr_times=10.0):
        self.base_lr = base_lr
        self.epochs = epochs
        self.iter_each_epoch = iters_each_epoch
        self.total_iters = self.epochs * self.iter_each_epoch
        self.power = power
        self.lr_times = lr_times

    def __call__(self, optimizer, iteration, epoch):
        lr = self.base_lr * (1 - iteration / self.total_iters) ** self.power
        optimizer.param_groups[0]["lr"] = lr
        for i in range(1, len(optimizer.param_groups)):
            optimizer.param_groups[i]["lr"] = lr * self.lr_times
