import os
import numpy as np
import torch
import shutil
import torchvision.transforms as transforms
from sklearn import metrics
from .solver import solve_isotropic_covariance, symKL_objective
import math
import ruamel.yaml as yaml
import torch.nn as nn


def save_model(model, model_path):
    torch.save(model.state_dict(), model_path)

def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path))

def save_checkpoint(state, is_best, save, checkpoint):
    filename = os.path.join(save, checkpoint)
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.sum_of_squares = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.sum_of_squares += (val ** 2) * n
        self.count += n
        self.avg = self.sum / self.count

    def std_dev(self):
        mean_of_squares = self.sum_of_squares / self.count
        square_of_mean = self.avg ** 2
        variance = mean_of_squares - square_of_mean
        return variance ** 0.5


def apply_noise_patch(noise,images,offset_x=0,offset_y=0,mode='change',padding=20,position='fixed'):
    '''
    noise: torch.Tensor(1, 3, pat_size, pat_size)
    images: torch.Tensor(N, 3, 512, 512)
    outputs: torch.Tensor(N, 3, 512, 512)
    '''
    length = images.shape[2] - noise.shape[2]
    if position == 'fixed':
        wl = offset_x
        ht = offset_y
    else:
        wl = np.random.randint(padding,length-padding)
        ht = np.random.randint(padding,length-padding)
    if images.dim() == 3:
        noise_now = noise.clone()[0,:,:,:]
        wr = length-wl
        hb = length-ht
        m = nn.ZeroPad2d((wl, wr, ht, hb))
        if(mode == 'change'):
            images[:,ht:ht+noise.shape[2],wl:wl+noise.shape[3]] = 0
            images += m(noise_now)
        else:
            images += noise_now
    else:
        for i in range(images.shape[0]):
            noise_now = noise.clone()
            wr = length-wl
            hb = length-ht
            m = nn.ZeroPad2d((wl, wr, ht, hb))
            if(mode == 'change'):
                images[i:i+1,:,ht:ht+noise.shape[2],wl:wl+noise.shape[3]] = 0
                images[i:i+1] += m(noise_now)
            else:
                images[i:i+1] += noise_now
    return images


def backdoor_truepostive_rate(confusion_matrix):
    tn, fp, fn, tp = confusion_matrix.ravel()
    poisoned_fn_rate = fn/(fn+tp)
    poisoned_tn_rate = tn/(tn+fp)
    poisoned_fp_rate = fp/(fp+tn)
    poisoned_tp_rate = tp/(tp+fn)

    return poisoned_fn_rate, poisoned_tn_rate, poisoned_fp_rate, poisoned_tp_rate

def compute_correct_prediction(*, y_targets, y_prob_preds, threshold=0.5):
    y_hat_lbls = []
    pred_pos_count = 0
    pred_neg_count = 0
    correct_count = 0
    for y_prob, y_t in zip(y_prob_preds, y_targets):
        if y_prob <= threshold:
            pred_neg_count += 1
            y_hat_lbl = 0
        else:
            pred_pos_count += 1
            y_hat_lbl = 1
        y_hat_lbls.append(y_hat_lbl)
        if y_hat_lbl == y_t:
            correct_count += 1

    return np.array(y_hat_lbls), [pred_pos_count, pred_neg_count, correct_count]

def gradient_masking(g):
    # add scalar noise to align with the maximum norm in the batch
    # (expectation norm alignment)
    # yjr:这里的输入不明确但可以确定的是g[0][0]的shape是[batch_size, dim]

    g_norm = torch.norm(g, p=2, dim=1)
    max_norm = torch.max(g_norm)
    stds = torch.sqrt(torch.maximum(max_norm ** 2 /
                              (g_norm ** 2 + 1e-32) - 1.0, torch.tensor(0.0)))
    standard_gaussian_noise = torch.normal(mean=0.0, std=1.0, size=(g_norm.shape[0], 1)).cuda()
    gaussian_noise = standard_gaussian_noise * stds.view(-1, 1)
    # res = [g[0][0] * (1 + gaussian_noise)]
    return g * (1 + gaussian_noise)

def gradient_gaussian_noise_masking(g, ratio):
    g_norm = torch.norm(g, p=2, dim=1)
    max_norm = torch.max(g_norm)
    gaussian_std = ratio * max_norm/torch.sqrt(torch.tensor(g.shape[1], dtype=torch.float))
    gaussian_noise = torch.normal(mean=0.0, std=gaussian_std, size=g.shape).cuda()
    # res = [g[0][0]+gaussian_noise]
    return g + gaussian_noise

def marvell_g(g, labels):
    print(g.shape)
    print(labels.shape)
    print(labels)
    y = labels
    pos_g = g[y[:, 0] == 1]
    pos_g_mean = torch.mean(pos_g, dim=0, keepdim=True) # shape [1, d]
    pos_coordinate_var = torch.mean(torch.square(pos_g-pos_g_mean), dim=0)

    neg_g = g[y[:, 0] == 0]
    neg_g_mean = torch.mean(neg_g, dim=0, keepdim=True)
    neg_coordinate_var = torch.mean(torch.square(neg_g-neg_g_mean), dim=0)

    avg_pos_coordinate_var = torch.mean(pos_coordinate_var)
    avg_neg_coordinate_var = torch.mean(neg_coordinate_var)

    g_diff = pos_g_mean - neg_g_mean

    g_diff_norm = float(torch.norm(g_diff))

    u = float(avg_neg_coordinate_var)
    v = float(avg_pos_coordinate_var)

    d = float(g.shape[1])

    p = float(torch.sum(y)/len(y))

    scale = 1.0

    lam10, lam20, lam11, lam21 = None, None, None, None


    P = scale * g_diff_norm ** 2
    # print('g_diff_norm ** 2', g_diff_norm ** 2)
    # print('P', P)
    # print('u, v, d, p, g_diff\n', u, v, d, p, g_diff)
    lam10, lam20, lam11, lam21, sumKL = \
        solve_isotropic_covariance(
            u=u,
            v=v,
            d=d,
            g=g_diff_norm ** 2,
            p=p,
            P=P,
            lam10_init=lam10,
            lam20_init=lam20,
            lam11_init=lam11,
            lam21_init=lam21)
        # print('sumKL', sumKL)
        # print()

        # print(scale)
        # if not dynamic or sumKL <= sumKL_threshold:

    if sumKL == -1:
        return [g]



    perturbed_g = g
    #y_float = torch.FloatTensor(y)

    # positive examples add noise in g1 - g0=
    perturbed_g += torch.mul(torch.randn(y.shape).cuda(), y) * g_diff * (math.sqrt(lam11 - lam21) / g_diff_norm)

    # add spherical noise to positive examples
    if lam21 > 0.0:
        perturbed_g += torch.randn(g.shape).cuda() * y * math.sqrt(lam21)

    # negative examples add noise in g1 - g0
    perturbed_g += torch.mul(torch.randn(y.shape).cuda(),
                                                1 - y) * g_diff * (
                           math.sqrt(lam10 - lam20) / g_diff_norm)

    # add spherical noise to negative examples
    if lam20 > 0.0:
        perturbed_g += torch.randn(g.shape).cuda() * (1 - y) * math.sqrt(lam20)

    return [perturbed_g]


def gradient_compression(g, preserved_perc=0.25):
    # 这里的输入不明确但可以确定的是g[0][0]的shape是[batch_size, dim]
    # 这里模仿了之前的函数
    # tensor = g[0][0]   #这里的g是一个历史遗留问题
    tensor = g
    tensor_copy = tensor.clone().detach()
    tensor_copy = torch.abs(tensor_copy)
    survivial_values = torch.topk(tensor_copy.reshape(1, -1),
                                  int(tensor_copy.reshape(1, -1).shape[1] * preserved_perc))
    # 这里取保留元素的最小值
    thresh_hold = survivial_values[0][0][-1]

    background_tensor = torch.zeros(tensor.shape).to(torch.float)
    if 'cuda' in str(tensor.device):
        background_tensor = background_tensor.cuda()
    # print("background_tensor", background_tensor)
    tensor = torch.where(abs(tensor) > thresh_hold, tensor, background_tensor)
    # print("tensor:", tensor)
    return tensor
def noisy_count(beta):
    # beta = sensitivity / epsilon
    beta = beta
    u1 = np.random.random()
    u2 = np.random.random()
    if u1 <= 0.5:
        n_value = -beta * np.log(1. - u2)
    else:
        n_value = beta * np.log(u2)
    n_value = torch.tensor(n_value)
    # print(n_value)
    return n_value

def laplacian_noise_masking(g, beta=0.001):
    # generate noisy mask
    # whether the tensor to process is on cuda devices
    noisy_mask = torch.zeros(g.shape).to(torch.float)
    if 'cuda' in str(g.device):
        noisy_mask = noisy_mask.cuda()
    noisy_mask = noisy_mask.flatten()
    for i in range(noisy_mask.shape[0]):
        noisy_mask[i] = noisy_count(beta)
    noisy_mask = noisy_mask.reshape(g.shape)
    # print("noisy_tensor:", noisy_mask)
    g = g + noisy_mask
    return g

def keep_predict_loss(y_true, y_pred):
    # print("y_true:", y_true)
    # print("y_pred:", y_pred[0][:5])
    # print("y_true * y_pred:", (y_true * y_pred))
    return torch.sum(y_true * y_pred)

def over_write_args_from_file(args, yml):
    """
    overwrite arguments according to config file
    """
    if yml == '':
        return
    with open(yml, 'r', encoding='utf-8') as f:
        dic = yaml.load(f.read(), Loader=yaml.Loader)
        for k in dic:
            setattr(args, k, dic[k])

def image_format_2_rgb(x):
    return x.convert("RGB")