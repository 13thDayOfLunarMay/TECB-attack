import logging
import os
import random
import sys

import numpy as np
from sklearn.utils import shuffle

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
from fedml_core.data_preprocessing.NUS_WIDE.nus_wide_dataset import NUS_WIDE_load_two_party_data
from fedml_core.data_preprocessing.NUS_WIDE.dataset import nuswide_dataset
from fedml_core.model.baseline.vfl_models import BottomModelForCifar100, TopModelForCifar100
from fedml_core.trainer.vfl_trainer import VFLTrainer
from fedml_core.utils.utils import AverageMeter, keep_predict_loss, over_write_args_from_file
from fedml_core.data_preprocessing.cifar100.dataset import IndexedCIFAR100

#from fedml_api.utils.utils import save_checkpoint
import torch
import torch.nn as nn
import argparse
import time
import glob
import shutil
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100
from torch.utils.data import Subset
import logging


def save_checkpoint(state, is_best, save, checkpoint):
    filename = os.path.join(save, checkpoint)
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)



def main(device, args):

    for seed in range(1):
        # random seed for 10 runs
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


        #load data
        # Data normalization and augmentation (optional)
        transform = transforms.Compose([
            #transforms.RandomHorizontalFlip(),
            #transforms.RandomCrop(32, padding=4),
            transforms.ToTensor()
            #transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])

        # Load CIFAR-10 dataset
        trainset = IndexedCIFAR100(root='./data', train=True, download=True, transform=transform)
        testset = IndexedCIFAR100(root='./data', train=False, download=True, transform=transform)

        # 选择你感兴趣的类别
        target_class = args.target_class

        target_label =trainset.class_to_idx[target_class]

        # 找出所有属于这个类别的样本的索引
        target_indices = np.where(np.array(trainset.targets) == target_label)[0]

        non_target_indices = np.where(np.array(testset.targets) != target_label)[0]

        non_target_set = Subset(testset, non_target_indices)

        # 从目标索引中随机选择10个索引
        selected_indices = np.random.choice(target_indices, args.poison_num, replace=False)

        train_queue = torch.utils.data.DataLoader(
            dataset=trainset,
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers
        )
        test_queue = torch.utils.data.DataLoader(
            dataset=testset,
            batch_size=args.batch_size,
            num_workers=args.workers
        )
        non_target_queue = torch.utils.data.DataLoader(
            dataset=non_target_set,
            batch_size=args.batch_size,
            num_workers=args.workers
        )


        model_list = []
        model_list.append(BottomModelForCifar100())
        model_list.append(BottomModelForCifar100())
        model_list.append(TopModelForCifar100())

        criterion = nn.CrossEntropyLoss().to(device)
        bottom_criterion = keep_predict_loss

        # 加载预训练模型

        save_model_dir = args.save + "/0_saved_models"
        checkpoint_path = save_model_dir + "/model_best.pth.tar"
        checkpoint = torch.load(checkpoint_path)

        # 加载每个模型的参数
        for i in range(len(model_list)):
            model_list[i].load_state_dict(checkpoint['state_dict'][i])

        # 加载delta,用来生成有毒样本
        delta = torch.load(save_model_dir + "/delta.pth")

        vfltrainer = VFLTrainer(model_list)

        print("################################ Test Backdoor Models ############################")

        test_loss, top1_acc, top5_acc = vfltrainer.test_mul(test_queue, criterion, device, args)
        _, test_asr_acc, test_asr_acc5 = vfltrainer.test_backdoor_mul(non_target_queue, criterion, device, args, delta,
                                                                      target_label)

        print("test_loss: ", test_loss, "top1_acc: ", top1_acc,
              "top5_acc: ", top5_acc, "test_asr_acc: ", test_asr_acc, "test_asr_acc5: ", test_asr_acc5)


if __name__ == '__main__':
    print("################################ Prepare Data ############################")

    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser("vflmodelnet")
    parser.add_argument('--data_dir', default="./data/CIFAR10/", help='location of the data corpus')
    parser.add_argument('-d', '--dataset', default='CIFAR100', type=str,
                        help='name of dataset',
                        choices=['CIFAR10', 'CIFAR100', 'TinyImageNet', 'CINIC10L', 'Yahoo', 'Criteo', 'BCW'])
    parser.add_argument('--name', type=str, default='vfl_cifar100', help='experiment name')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--lr', type=float, default=0.1, help='init learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--report_freq', type=float, default=10, help='report frequency')
    parser.add_argument('--workers', type=int, default=0, help='num of workers')
    parser.add_argument('--epochs', type=int, default=100, help='num of training epochs')
    parser.add_argument('--layers', type=int, default=18, help='total number of layers')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--grad_clip', type=float, default=5., help='gradient clipping')
    parser.add_argument('--gamma', type=float, default=0.97, help='learning rate decay')
    parser.add_argument('--decay_period', type=int, default=1, help='epochs between two learning rate decays')
    parser.add_argument('--parallel', action='store_true', default=False, help='data parallelism')
    parser.add_argument('--u_dim', type=int, default=64, help='u layer dimensions')
    parser.add_argument('--k', type=int, default=2, help='num of client')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--save', default='./model/CIFAR100/baseline', type=str,
                        metavar='PATH',
                        help='path to save checkpoint (default: none)')
    parser.add_argument('--step_gamma', default=0.1, type=float, metavar='S',
                        help='gamma for step scheduler')
    parser.add_argument('--stone1', default=50, type=int, metavar='s1',
                        help='stone1 for step scheduler')
    parser.add_argument('--stone2', default=85, type=int, metavar='s2',
                        help='stone2 for step scheduler')
    parser.add_argument('--half', help='half number of features, generally seen as the adversary\'s feature num. '
                                       'You can change this para (lower that party_num) to evaluate the sensitivity '
                                       'of our attack -- pls make sure that the model to be resumed is '
                                       'correspondingly trained.',
                        type=int,
                        default=16)  # choices=[16, 14, 32, 1->party_num]. CIFAR10-16, Liver-14, TinyImageNet-32

    parser.add_argument('--marvell', action='store_true', default=False, help='marvell defense')
    parser.add_argument('--max_norm', action='store_true', default=False, help='maxnorm defense')
    parser.add_argument('--iso', action='store_true', default=False, help='iso defense')
    parser.add_argument('--gc', action='store_true', default=False, help='gc defense')
    parser.add_argument('--lap_noise', action='store_true', default=False, help='lap_noise defense')
    parser.add_argument('--signSGD', action='store_true', default=False, help='sign_SGD defense')

    parser.add_argument('--iso_ratio', type=float, default=0.01, help='iso defense ratio')
    parser.add_argument('--gc_ratio', type=float, default=0.01, help='gc defense ratio')
    parser.add_argument('--lap_noise_ratio', type=float, default=0.01, help='lap_noise defense ratio')

    parser.add_argument('--poison_num', type=int, default=100, help='num of poison data')
    parser.add_argument('--corruption_amp', type=float, default=10, help='amplication of corruption')
    parser.add_argument('--backdoor_start', action='store_true', default=False, help='backdoor')
    # config file
    parser.add_argument('--c', type=str, default='configs/base/cifar100_bestattack.yml')

    args = parser.parse_args()
    over_write_args_from_file(args, args.c)

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    # 创建一个logger
    logger = logging.getLogger('experiment_logger')
    logger.setLevel(logging.INFO)

    # 创建一个handler，用于写入日志文件
    fh = logging.FileHandler(args.save +'/experiment.log')
    fh.setLevel(logging.INFO)

    # 定义handler的输出格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)

    # 给logger添加handler
    logger.addHandler(fh)

    logger.info(args)
    logger.info(device)


    main(device=device, args=args)
    # reference training result:
    # --- epoch: 99, batch: 1547, loss: 0.11550658332804839, acc: 0.9359105089400196, auc: 0.8736984159409958
    # --- (0.9270889578726378, 0.5111934752243287, 0.5054099033579607, None)

    # --- epoch: 99, batch: 200, loss: 0.09191526211798191, acc: 0.9636565918783608, auc: 0.9552342451916291
    # --- (0.9754657898538487, 0.7605652456769234, 0.8317858679682943, None)

