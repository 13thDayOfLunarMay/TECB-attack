import logging
import os
import random
import sys

import numpy as np
from sklearn.utils import shuffle

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
from fedml_core.data_preprocessing.NUS_WIDE.nus_wide_dataset import NUS_WIDE_load_two_party_data
from fedml_core.data_preprocessing.cifar10.dataset import IndexedCIFAR10
from fedml_core.model.baseline.vfl_models import BottomModelForCifar10, TopModelForCifar10
from fedml_core.trainer.vfl_trainer import VFLTrainer
from fedml_core.utils.utils import AverageMeter, keep_predict_loss, over_write_args_from_file

# from fedml_api.utils.utils import save_checkpoint
import torch
import torch.nn as nn
import argparse
import time
import glob
import shutil
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
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

        # load data
        # Data normalization and augmentation (optional)
        train_transform = transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465) , (0.2471, 0.2435, 0.2616))
        ])

        # Load CIFAR-10 dataset
        # 唯一的区别是，IndexedCIFAR10 类返回的图片的第三个元素是图片的索引
        trainset = IndexedCIFAR10(root='./data', train=True, download=True, transform=train_transform)
        testset = IndexedCIFAR10(root='./data', train=False, download=True, transform=train_transform)

        # CIFAR-10 类别标签（以类别名称的列表形式给出）
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        # 选择你感兴趣的类别
        target_class = args.target_class

        # 找出这个类别的标签
        target_label = classes.index(target_class)

        # 找出所有属于这个类别的样本的索引

        non_target_indices = np.where(np.array(testset.targets) != target_label)[0]

        non_target_set = Subset(testset, non_target_indices)

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


        # build model
        model_list = []
        model_list.append(BottomModelForCifar10())
        model_list.append(BottomModelForCifar10())
        model_list.append(TopModelForCifar10())

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

        test_loss, test_asr_acc, _ = vfltrainer.test_backdoor_mul(non_target_queue, criterion, device, args, delta,
                                                                  target_label)
        print("test_loss: ", test_loss, "top1_acc: ", top1_acc,
              "top5_acc: ", top5_acc, "test_asr_acc: ", test_asr_acc)




if __name__ == '__main__':
    print("################################ Prepare Data ############################")

    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    parser = argparse.ArgumentParser("vflmodelnet")

    parser.add_argument('--data_dir', default="./data/CIFAR10/", help='location of the data corpus')
    parser.add_argument('-d', '--dataset', default='CIFAR10', type=str,
                        help='name of dataset',
                        choices=['CIFAR10', 'CIFAR100', 'TinyImageNet', 'CINIC10L', 'Yahoo', 'Criteo', 'BCW'])
    parser.add_argument('--name', type=str, default='vfl_cifar10', help='experiment name')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--lr', type=float, default=0.1, help='init learning rate')
    parser.add_argument('--trigger_lr', type=float, default=0.001, help='init learning rate')
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
    parser.add_argument('--save', default='./model/CIFAR10/baseline', type=str,
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
    parser.add_argument('--backdoor', type=float, default=20, help='backdoor frequency')
    parser.add_argument('--poison_epochs', type=float, default=20, help='backdoor frequency')
    parser.add_argument('--target_class', type=str, default='cat', help='backdoor target class')
    parser.add_argument('--alpha', type=float, default=0.01, help='uap learning rate decay')
    parser.add_argument('--eps', type=float, default=16 / 255, help='uap clamp bound')

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
    parser.add_argument('--c', type=str, default='configs/base/cifar10_bestattack.yml', help='config file')

    args = parser.parse_args()
    over_write_args_from_file(args, args.c)

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    # 创建一个logger
    logger = logging.getLogger('experiment_logger')
    logger.setLevel(logging.INFO)

    # 创建一个handler，用于写入日志文件
    fh = logging.FileHandler(args.save + '/experiment.log')
    fh.setLevel(logging.INFO)

    # 定义handler的输出格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)

    # 给logger添加handler
    logger.addHandler(fh)

    logger.info(args)
    logger.info(device)

    main(device=device, args=args)

    # Test set: Average loss: 0.0110, Top-1 Accuracy: 8052.0/10000 (80.5200%), Top-5 Accuracy: 9787.0/10000 (97.8700%)
    # Backdoor Test set: Average loss: 0.6695, ASR Top-1 Accuracy: 9000.0/9000 (100.0000%, ASR Top-5 Accuracy: 9000.0/9000 (100.0000%)
    # test_loss:  0.01103087488412857 top1_acc:  80.52 top5_acc:  97.87 test_asr_acc:  100.0

