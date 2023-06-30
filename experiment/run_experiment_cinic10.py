import argparse
import torch
from multiprocessing import Pool
from vfl_cinic10_training import *
from concurrent.futures import ProcessPoolExecutor

def set_args(parser):
    parser = argparse.ArgumentParser("vflmodelnet")
    parser.add_argument('--data_dir', default="./data/CINIC-10/", help='location of the data corpus')
    parser.add_argument('-d', '--dataset', default='CINIC10L', type=str,
                        help='name of dataset',
                        choices=['CIFAR10', 'CIFAR100', 'TinyImageNet', 'CINIC10L', 'Yahoo', 'Criteo', 'BCW'])
    parser.add_argument('--name', type=str, default='vfl_CINIC10L', help='experiment name')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--lr', type=float, default=0.1, help='init learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--report_freq', type=float, default=10, help='report frequency')
    parser.add_argument('--workers', type=int, default=1, help='num of workers')
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
    parser.add_argument('--save', default='./model/CINIC10/test/backdoor50_LRA_poison4_amp15', type=str,
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
    parser.add_argument('--c', type=str, default='configs/noaug/cinic10_bestattack.yml', help='config file')

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
    torch.backends.cudnn.enabled = False
    # 列出所有的防御方法
    # protectMethod = ['non', 'max_norm', 'iso', 'gc', 'lap_noise', 'signSGD']
    protectMethod = ['non', 'max_norm', 'iso', 'gc', 'signSGD']

    # protectMethod = ['signSGD']

    iso_range = [0.001, 0.01, 0.1, 0.5, 1.0]
    gc_range = [0.01, 0.1, 0.25, 0.5, 0.75]
    lap_noise_range = [0.0001, 0.001, 0.01, 0.05, 0.1]

    list_of_args = []

    save_path = './model/CINIC10/noaug/defense/'

    # 总共有18个实验组合
    for method in protectMethod:
        if method == 'non':
            parser = argparse.ArgumentParser("vflmodelnet")
            args = set_args(parser)
            args.save = save_path + 'non'
            list_of_args.append(args)
        elif method == 'max_norm':
            parser = argparse.ArgumentParser("vflmodelnet")
            args = set_args(parser)
            args.max_norm = True
            args.save = save_path + 'max_norm'
            list_of_args.append(args)
        elif method == 'iso':
            for iso in iso_range:
                parser = argparse.ArgumentParser("vflmodelnet")
                args = set_args(parser)
                args.iso = True
                args.iso_ratio = iso
                args.save = save_path + 'iso' + str(iso)
                list_of_args.append(args)
        elif method == 'gc':
            for gc in gc_range:
                parser = argparse.ArgumentParser("vflmodelnet")
                args = set_args(parser)
                args.gc = True
                args.gc_ratio = gc
                args.save = save_path + 'gc' + str(gc)
                list_of_args.append(args)
        elif method == 'lap_noise':
            for lap_noise in lap_noise_range:
                parser = argparse.ArgumentParser("vflmodelnet")
                args = set_args(parser)
                args.lap_noise = True
                args.lap_noise_ratio = lap_noise
                args.save = save_path + 'lap_noise' + str(lap_noise)
                list_of_args.append(args)
        elif method == 'signSGD':
            parser = argparse.ArgumentParser("vflmodelnet")
            args = set_args(parser)
            args.signSGD = True
            args.save = save_path + 'signSGD'
            list_of_args.append(args)
    # print(len(list_of_args))





    # Create a pool of workers and run experiments in parallel
    # 同时最大运行3个进程
    # with Pool(processes=3) as pool:
    #     pool.map(run_experiment, list_of_args)
    with ProcessPoolExecutor(max_workers=3) as executor:
        executor.map(run_experiment, list_of_args)
