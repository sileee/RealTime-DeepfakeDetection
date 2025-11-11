import os
import sys
import time
import torch
import torch.nn
import argparse
# from tensorboardX import SummaryWriter # 保持注释，若未安装则不导入
import numpy as np
from validate import validate
from data import create_dataloader
from networks.trainer import Trainer
from options.train_options import TrainOptions
from options.test_options import TestOptions
from util import Logger
import torchvision
from earlystop import EarlyStopping
import random
from pathlib import Path


# --- 辅助函数 ---

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    print(f"seed: {seed}")


def get_val_opt(original_dataroot, val_split):
    """
    根据原始训练选项创建一个用于验证的选项副本。
    """
    # 使用 TrainOptions 解析，以继承训练所需的所有参数
    val_opt = TrainOptions().parse(print_options=False)

    # 继承主要参数
    val_opt.dataroot = original_dataroot
    val_opt.val_split = val_split  # 'test'
    val_opt.name = 'VAL_TEMP'

    # 设置验证模式的通用参数
    val_opt.isTrain = False
    val_opt.no_resize = False
    val_opt.no_crop = False
    val_opt.serial_batches = True

    return val_opt


if __name__ == '__main__':
    opt = TrainOptions().parse()
    set_seed()

    # 1. 保存原始的 dataroot 和 val_split
    original_dataroot = opt.dataroot  # datasets/WildRF
    original_val_split = opt.val_split  # test

    # 2. 设置训练集路径：opt.dataroot 现在指向 datasets/WildRF/train
    opt.dataroot = os.path.join(opt.dataroot, opt.train_split)

    Logger(os.path.join(opt.checkpoints_dir, opt.name, 'log.log'))
    print('  '.join(list(sys.argv)))

    # 3. 准备训练数据加载器
    train_data_loader, train_paths = create_dataloader(opt)
    dataset_size = len(train_data_loader)
    print('#training images = %d' % dataset_size)

    # 4. 准备验证数据选项 (使用 REDDIT 平台作为验证集)
    val_opt = get_val_opt(original_dataroot, original_val_split)

    # ❗️ 核心修正：val_opt.dataroot 必须精确指向二分类目录
    # WildRF/test/reddit 是最合适的验证集，它下面有 0_real 和 1_fake
    val_opt.dataroot = os.path.join(val_opt.dataroot, val_opt.val_split, 'reddit')

    # 检查验证路径
    if not os.path.isdir(val_opt.dataroot):
        print(f"❌ Validation Data Path Missing: {val_opt.dataroot}")
        raise FileNotFoundError(f"Please check if {val_opt.dataroot} contains '0_real' and '1_fake'.")

    # 5. 初始化模型和训练器
    # (假设 SummaryWriter 和 wandb 导入已移除)
    # train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "train"))
    # val_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "val"))
    model = Trainer(opt)
    net_params = sum(map(lambda x: x.numel(), model.parameters()))
    print(f'Model parameters {net_params:,d}')

    # initializing stopping criteria
    early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.001, verbose=True)

    print(f'cwd: {os.getcwd()}')
    # training loop
    for epoch in range(opt.niter):
        model.train()
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        # 训练迭代
        for i, data in enumerate(train_data_loader):
            model.total_steps += 1
            epoch_iter += opt.batch_size
            model.set_input(data)
            model.optimize_parameters()

            if model.total_steps % opt.loss_freq == 0:
                # 假设 train_writer 导入被注释，我们只打印
                print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()),
                      "Train loss: {} at step: {} lr {}".format(model.loss, model.total_steps, model.lr))

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, model.total_steps))
            model.save_networks('latest')
            model.save_networks(epoch)

        # 6. 验证 (每一轮都进行)
        model.eval()
        # validate(model.model, val_opt) 现在使用 datasets/WildRF/test/reddit 进行二分类验证
        acc, ap = validate(model.model, val_opt)[:2]

        # 假设 val_writer 导入被注释
        # val_writer.add_scalar('accuracy', acc, model.total_steps)
        # val_writer.add_scalar('ap', ap, model.total_steps)
        print("(Val @ epoch {}) acc: {}; ap: {}".format(epoch, acc, ap))
        model.train()

        # 7. 早停逻辑
        # wandb.log 相关的代码被移除，因为可能没有安装 wandb
        early_stopping(acc, model)
        if early_stopping.early_stop:
            cont_train = model.adjust_learning_rate()
            if cont_train:
                print("Learning rate dropped by 10, continue training...")
                # 重置 EarlyStopping 的耐心值 (patience)
                early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.002, verbose=True)
            else:
                print("Early stopping.")
                break

    model.save_networks('last')