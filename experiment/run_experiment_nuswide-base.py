import argparse
import torch
from multiprocessing import Pool
from vfl_nuswide_training import *


def set_args(parser):
    parser.add_argument('--data_dir', default="./data/NUS_WIDE/", help='location of the data corpus')
    parser.add_argument('--name', type=str, default='vfl_nus_baseline', help='experiment name')
    parser.add_argument('-d', '--dataset', default='NUS_WIDE', type=str,
                        help='name of dataset',
                        choices=['CIFAR10', 'CIFAR100', 'TinyImageNet', 'CINIC10L', 'Yahoo', 'Criteo', 'BCW',
                                 'NUS_WIDE'])
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--lr', type=float, default=0.1, help='init learning rate')
    parser.add_argument('--trigger_lr', type=float, default=0.001, help='init learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='weight decay')
    parser.add_argument('--report_freq', type=float, default=10, help='report frequency')
    parser.add_argument('--workers', type=int, default=0, help='num of workers')
    parser.add_argument('--epochs', type=int, default=20, help='num of training epochs')
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
    parser.add_argument('--save', default='./model/nuswide/baseline_smooth', type=str,
                        metavar='PATH',
                        help='path to save checkpoint (default: none)')
    parser.add_argument('--step_gamma', default=0.1, type=float, metavar='S',
                        help='gamma for step scheduler')
    parser.add_argument('--stone1', default=15, type=int, metavar='s1',
                        help='stone1 for step scheduler')
    parser.add_argument('--stone2', default=40, type=int, metavar='s2',
                        help='stone2 for step scheduler')

    parser.add_argument('--half', help='half number of features, generally seen as the adversary\'s feature num. '
                                       'You can change this para (lower that party_num) to evaluate the sensitivity '
                                       'of our attack -- pls make sure that the model to be resumed is '
                                       'correspondingly trained.',
                        type=int,
                        default=16)  # choices=[16, 14, 32, 1->party_num]. CIFAR10-16, Liver-14, TinyImageNet-32
    parser.add_argument('--backdoor', type=float, default=20, help='backdoor frequency')
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
    parser.add_argument('--c', type=str, default='configs/Narcissus/nuswide_bestattack.yml', help='config file')

    args = parser.parse_args()
    over_write_args_from_file(args, args.c)

    return args



def run_experiment(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("################################ start experiment ############################")
    print(args.save)
    print(device)

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    txt_name = f"saved_process_data"
    savedStdout = sys.stdout
    with open(args.save + '/' + txt_name + '.txt', 'a') as file:
        sys.stdout = file
        main(device=device, args=args)
        sys.stdout = savedStdout
    print("################################ end experiment ############################")


if __name__ == '__main__':
    # 列出所有的防御方法
    eplison_range = [0.1, 0.5, 1.0, 2.0]
    poison_num_range = [1, 2, 5, 10]

    list_of_args = []

    save_path = './model/nuswide/multiseeds/'

    for epl in eplison_range:
        parser = argparse.ArgumentParser("vflmodelnet")
        args = set_args(parser)
        args.eps = epl
        args.save = save_path + 'eplison' + str(epl)
        list_of_args.append(args)

    for num in poison_num_range:
        parser = argparse.ArgumentParser("vflmodelnet")
        args = set_args(parser)
        args.poison_num = num
        args.save = save_path + 'poison_num' + str(num)
        list_of_args.append(args)




    # Create a pool of workers and run experiments in parallel
    # 同时最大运行3个进程
    with Pool(processes=3) as pool:
        pool.map(run_experiment, list_of_args)

