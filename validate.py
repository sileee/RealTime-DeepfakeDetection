import torch
import numpy as np
from networks.LaDeDa import LaDeDa9, LaDeDa
from networks.Tiny_LaDeDa import tiny_ladeda
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score, roc_auc_score, \
    precision_score, recall_score
from data import create_dataloader
import torch.nn as nn

from options.test_options import TestOptions


def validate(model, opt):
    data_loader, _ = create_dataloader(opt)
    with torch.no_grad():
        y_true, y_pred = [], []
        for img, label in data_loader:
            img_input = img.cuda()
            y_pred.extend(model(img_input).sigmoid().flatten().tolist())
            # ✅ 直接使用原始标签（不再做 % 2 映射）
            y_true.extend(label.flatten().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    if len(y_true[y_true == 0]) == 0:
        r_acc = 0.0
    else:
        r_acc = accuracy_score(y_true[y_true == 0], y_pred[y_true == 0] > 0.5)

    if len(y_true[y_true == 1]) == 0:
        f_acc = 0.0
    else:
        f_acc = accuracy_score(y_true[y_true == 1], y_pred[y_true == 1] > 0.5)

    acc = accuracy_score(y_true, y_pred > 0.5)
    ap = average_precision_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred > 0.5, zero_division=0)
    recall = recall_score(y_true, y_pred > 0.5, zero_division=0)

    return acc, ap, r_acc, f_acc, auc, precision, recall


if __name__ == '__main__':
    opt = TestOptions().parse(print_options=False)
    model = LaDeDa(num_classes=1)
    state_dict = torch.load(opt.model_path, map_location='cpu')
    model.load_state_dict(state_dict['model'])
    model.cuda()
    model.eval()

    acc, ap, r_acc, f_acc, auc, precision, recall = validate(model, opt)

    print("accuracy:", acc)
    print("average precision:", ap)  # ✅ 修正：avg_precision → ap
    print("accuracy of real images:", r_acc)
    print("accuracy of fake images:", f_acc)