
import functools
import torch
import torch.nn as nn
# from networks.LaDeDa import LaDeDa9
from networks.base_model import BaseModel, init_weights
from networks.LaDeDa_SoftAttention import LaDeDa9,LaDeDa_SoftAttention


class Trainer(BaseModel):
    def name(self):
        return 'Trainer'

    def __init__(self, opt):
        super(Trainer, self).__init__(opt)

        if self.isTrain and not opt.continue_train:
            # 旧代码测试，先注释
            # self.model = LaDeDa9(num_classes=1)
            # self.model.fc = nn.Linear(2048, 1)
            # torch.nn.init.normal_(self.model.fc.weight.data, 0.0, opt.init_gain)

            self.model = LaDeDa9(num_classes=1)
            print(f"当前模型类名: {self.model.__class__.__name__}", flush=True)

            if hasattr(self.model, 'final_fc'):
                final_fc_in_features = self.model.final_fc.in_features
                # 关键修改：添加 flush=True
                print(f"最终分类器输入特征维度: {final_fc_in_features}", flush=True)

                if final_fc_in_features == 4096:
                    print("✅ 确认：模型为 Soft Attention 版本 (4096 维度)。", flush=True)
                # ... (其余检查代码) ...
            # 初始化新模型的 *最终* 层
            torch.nn.init.normal_(self.model.final_fc.weight.data, 0.0, opt.init_gain)


        if not self.isTrain or opt.continue_train:
            # --- 删除旧代码 ---
            # self.model = LaDeDa9(pretrained=False, num_classes=1)

            # --- 添加新代码 (方案2) ---
            self.model = LaDeDa9(pretrained=False, num_classes=1)
            print(f"当前模型类名: {self.model.__class__.__name__}", flush=True)

            if hasattr(self.model, 'final_fc'):
                final_fc_in_features = self.model.final_fc.in_features
                # 关键修改：添加 flush=True
                print(f"最终分类器输入特征维度: {final_fc_in_features}", flush=True)

                if final_fc_in_features == 4096:
                    print("✅ 确认：模型为 Soft Attention 版本 (4096 维度)。", flush=True)
                # ... (其余检查代码) ...

        if self.isTrain:
            self.loss_fn = nn.BCEWithLogitsLoss()
            # initialize optimizers
            if opt.optim == "adam_cosine":
                self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                                  lr=opt.max_lr, betas=(opt.beta1, 0.999))
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=opt.niter, eta_min=opt.min_lr)
            elif opt.optim == 'adam':
                self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))
            elif opt.optim == 'sgd':
                self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),
                                                 lr=opt.lr, momentum=0.0, weight_decay=0)
            else:
                raise ValueError("optim should be [adam, sgd]")

        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.epoch)
        self.model.to(opt.gpu_ids[0])

    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= 0.9
            if param_group['lr'] < min_lr:
                return False
        self.lr = param_group['lr']
        print('*' * 25)
        print(f'Changing lr from {param_group["lr"] / 0.9} to {param_group["lr"]}')
        print('*' * 25)
        return True


    def set_input(self, input):
        self.input = input[0].to(self.device)
        self.label = input[1].to(self.device).float()

    def forward(self):
        self.output = self.model(self.input)

    def get_loss(self):
        return self.loss_fn(self.output.squeeze(1), self.label)

    def optimize_parameters(self):
        self.forward()
        self.loss = self.loss_fn(self.output.squeeze(1), self.label)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
