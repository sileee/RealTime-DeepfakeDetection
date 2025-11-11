# test.py
import os
import torch
import torch.nn as nn
import numpy as np
import random
from pathlib import Path
from collections import OrderedDict
from copy import deepcopy

from options.test_options import TestOptions
from validate import validate
from networks.LaDeDa import LaDeDa9
from networks.Tiny_LaDeDa import tiny_ladeda


# ==================== 设置随机种子 ====================
def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🌱 Random seed set to: {seed}")


# ==================== 加载模型 ====================
def load_model(model_path):
    """
    根据文件名自动判断模型类型并加载
    """
    model_name = Path(model_path).stem

    # 判断是否为 Tiny 模型
    is_tiny = 'Tiny' in model_name

    # 判断训练数据集
    trained_on = 'WildRF' if 'WildRF' in model_name else 'ForenSynth'

    print(f"\n{'=' * 60}")
    print(f"📦 Loading Model: {model_name}")
    print(f"   Type: {'Tiny-LaDeDa' if is_tiny else 'LaDeDa9'}")
    print(f"   Trained on: {trained_on}")
    print(f"{'=' * 60}")

    # 创建模型
    if is_tiny:
        features_dim = 8
        model = tiny_ladeda(num_classes=1, preprocess_type='NPR')
    else:
        features_dim = 2048
        model = LaDeDa9(num_classes=1, preprocess_type='NPR')

    # 加载权重
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model file not found: {model_path}")

    print(f"📥 Loading weights from: {model_path}")
    state_dict = torch.load(model_path, map_location='cpu')

    # 处理可能的格式
    if isinstance(state_dict, dict):
        if 'model' in state_dict:
            state_dict = state_dict['model']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

    # 清理state_dict的key
    cleaned_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace('module.', '') if k.startswith('module.') else k
        cleaned_state_dict[new_key] = deepcopy(v)

    # 加载到模型
    try:
        model.load_state_dict(cleaned_state_dict, strict=True)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"⚠️  Warning during loading: {e}")
        print("   Trying with strict=False...")
        model.load_state_dict(cleaned_state_dict, strict=False)

    model.eval()
    model.cuda()

    return model, features_dim, trained_on


# ==================== 测试WildRF ====================
def test_wildrf(model, opt):
    """
    测试WildRF数据集（Reddit, Facebook, Twitter）
    """
    print(f"\n{'=' * 60}")
    print("🧪 Testing on WildRF Dataset")
    print(f"{'=' * 60}")

    # ✅ 自动处理路径
    if hasattr(opt, 'dataroot') and opt.dataroot:
        dataroot = opt.dataroot
    else:
        dataroot = './datasets/WildRF/test'

    # 如果dataroot不以test结尾，自动添加
    if not dataroot.endswith('/test') and not dataroot.endswith('/test/'):
        if 'WildRF' in dataroot and not os.path.exists(os.path.join(dataroot, 'reddit')):
            dataroot = os.path.join(dataroot, 'test')

    print(f"📁 Data root: {dataroot}")

    platforms = ['reddit', 'facebook', 'twitter']
    results = {}
    accs, aps, aucs = [], [], []

    for platform in platforms:
        platform_path = os.path.join(dataroot, platform)

        if not os.path.exists(platform_path):
            print(f"⚠️  {platform} not found at {platform_path}, skipping...")
            continue

        print(f"\n📊 Evaluating {platform.upper()}...")

        # 设置验证参数
        opt.dataroot = platform_path
        opt.classes = ['']
        opt.no_resize = False
        opt.no_crop = True
        opt.is_aug = False

        try:
            acc, ap, r_acc, f_acc, auc, precision, recall = validate(model, opt)

            results[platform] = {
                'ACC': acc,
                'AP': ap,
                'AUC': auc,
                'Real_ACC': r_acc,
                'Fake_ACC': f_acc,
                'Precision': precision,
                'Recall': recall
            }

            accs.append(acc)
            aps.append(ap)
            aucs.append(auc)

            print(f"   ✅ ACC: {acc * 100:5.1f}% | AP: {ap * 100:5.1f}% | AUC: {auc * 100:5.1f}%")

        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 打印汇总
    if len(results) > 0:
        print(f"\n{'-' * 60}")
        print("📈 WildRF Results Summary")
        print(f"{'-' * 60}")
        print(f"{'Platform':<12} | {'ACC (%)':>8} | {'AP (%)':>8} | {'AUC (%)':>8}")
        print(f"{'-' * 60}")

        for platform, metrics in results.items():
            print(f"{platform.upper():<12} | "
                  f"{metrics['ACC'] * 100:>8.2f} | "
                  f"{metrics['AP'] * 100:>8.2f} | "
                  f"{metrics['AUC'] * 100:>8.2f}")

        print(f"{'-' * 60}")
        print(f"{'MEAN':<12} | "
              f"{np.mean(accs) * 100:>8.2f} | "
              f"{np.mean(aps) * 100:>8.2f} | "
              f"{np.mean(aucs) * 100:>8.2f}")
        print(f"{'=' * 60}")


# ==================== 测试ForenSynth ====================
def test_forensynth(model, opt):
    """
    测试ForenSynth数据集
    """
    print(f"\n{'=' * 60}")
    print("🧪 Testing on ForenSynth Dataset")
    print(f"{'=' * 60}")

    dataroot_forensynth = './datasets/CNNDetection/test'

    if not os.path.exists(dataroot_forensynth):
        print(f"❌ ForenSynth test data not found at {dataroot_forensynth}, skipping.")
        return {}

    print(f"📁 Data root: {dataroot_forensynth}")

    generators = ['progan', 'biggan', 'stylegan', 'stylegan2',
                  'cyclegan', 'stargan', 'gaugan', 'deepfake']

    results = {}
    accs, aps, aucs = [], [], []

    for gen in generators:
        gen_path = os.path.join(dataroot_forensynth, gen)

        if not os.path.exists(gen_path):
            print(f"⚠️  {gen} not found, skipping...")
            continue

        print(f"\n📊 Evaluating {gen.upper()}...")

        # ✅ 检查是否有嵌套结构
        subdirs = os.listdir(gen_path)
        has_binary_structure = '0_real' in subdirs and '1_fake' in subdirs

        if has_binary_structure:
            # 结构1: gen/0_real, gen/1_fake (BigGAN, StarGAN, GauGAN)
            print(f"   📁 Direct binary structure")
            opt.dataroot = gen_path
            opt.classes = ['']
            opt.no_resize = False
            opt.no_crop = True
            opt.is_aug = False

            try:
                acc, ap, r_acc, f_acc, auc, precision, recall = validate(model, opt)

                results[gen] = {'ACC': acc, 'AP': ap, 'AUC': auc}
                accs.append(acc)
                aps.append(ap)
                aucs.append(auc)

                print(f"   ✅ ACC: {acc * 100:5.1f}% | AP: {ap * 100:5.1f}%")
            except Exception as e:
                print(f"   ❌ Error: {e}")
                continue
        else:
            # 结构2: gen/category/0_real, gen/category/1_fake (ProGAN, StyleGAN, etc.)
            print(f"   📁 Nested category structure with {len(subdirs)} categories")

            # ✅ 对每个category分别测试，然后平均
            category_accs, category_aps = [], []

            for category in subdirs:
                category_path = os.path.join(gen_path, category)
                if not os.path.isdir(category_path):
                    continue

                # 检查是否有 0_real 和 1_fake
                if not (os.path.exists(os.path.join(category_path, '0_real')) and
                        os.path.exists(os.path.join(category_path, '1_fake'))):
                    continue

                opt.dataroot = category_path
                opt.classes = ['']
                opt.no_resize = False
                opt.no_crop = True
                opt.is_aug = False

                try:
                    acc, ap, r_acc, f_acc, auc, precision, recall = validate(model, opt)
                    category_accs.append(acc)
                    category_aps.append(ap)
                except Exception as e:
                    print(f"      ⚠️  Error on {category}: {e}")
                    continue

            if len(category_accs) > 0:
                avg_acc = np.mean(category_accs)
                avg_ap = np.mean(category_aps)

                results[gen] = {'ACC': avg_acc, 'AP': avg_ap, 'AUC': avg_acc}
                accs.append(avg_acc)
                aps.append(avg_ap)
                aucs.append(avg_acc)

                print(
                    f"   ✅ ACC: {avg_acc * 100:5.1f}% | AP: {avg_ap * 100:5.1f}% (avg over {len(category_accs)} categories)")

        # 打印汇总
    if len(results) > 0:
        print(f"\n{'-' * 60}")
        print("📈 ForenSynth Results Summary")
        print(f"{'-' * 60}")
        print(f"{'Generator':<14} | {'ACC (%)':>8} | {'AP (%)':>8}")
        print(f"{'-' * 60}")

        for gen, metrics in results.items():
            print(f"{gen.upper():<14} | "
                  f"{metrics['ACC'] * 100:>8.2f} | "
                  f"{metrics['AP'] * 100:>8.2f}")

        print(f"{'-' * 60}")
        print(f"{'MEAN':<14} | "
              f"{np.mean(accs) * 100:>8.2f} | "
              f"{np.mean(aps) * 100:>8.2f}")
        print(f"{'=' * 60}")


# ==================== 主函数 ====================
def main():
    # 设置随机种子
    set_seed(42)

    # 解析参数
    opt = TestOptions().parse(print_options=False)

    # 检查model_path
    if not hasattr(opt, 'model_path') or not opt.model_path:
        raise ValueError("❌ Please specify --model_path")

    # 加载模型
    model, features_dim, trained_on = load_model(opt.model_path)

    if hasattr(opt, 'dataset') and opt.dataset:
        test_on = opt.dataset.lower()  # ✅ 优先使用命令行参数
        print(f"📌 Using command line dataset: {test_on}")
    else:
        # 只有没有指定时才根据模型名判断
        test_on = 'wildrf' if 'WildRF' in opt.model_path else 'forensynth'
        print(f"📌 Auto-detected dataset: {test_on}")

    print(f"\n🎯 Test Configuration:")
    print(f"   Model: {Path(opt.model_path).name}")
    print(f"   Features Dim: {features_dim}")
    print(f"   Test Dataset: {test_on.upper()}")

    # 执行测试
    if test_on == 'wildrf':
        results = test_wildrf(model, opt)
    elif test_on == 'forensynth':
        results = test_forensynth(model, opt)
    elif test_on == 'both':
        print("\n🔄 Testing on both datasets...")
        wildrf_results = test_wildrf(model, opt)
        forensynth_results = test_forensynth(model, opt)
    else:
        raise ValueError(f"Unknown dataset: {test_on}")

    print("\n✅ Testing completed!")


if __name__ == '__main__':
    main()