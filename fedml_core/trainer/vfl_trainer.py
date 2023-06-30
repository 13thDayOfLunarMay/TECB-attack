import torch
from torch import nn
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support, classification_report, \
    confusion_matrix, roc_curve
import numpy as np
# import wandb
import torch.nn.functional as F
# from fedml_core.utils.utils import AverageMeter, gradient_masking, gradient_gaussian_noise_masking, marvell_g, backdoor_truepostive_rate, apply_noise_patch, gradient_compression, laplacian_noise_masking
from fedml_core.utils.utils import AverageMeter, gradient_masking, gradient_gaussian_noise_masking, marvell_g, \
    backdoor_truepostive_rate, gradient_compression, laplacian_noise_masking
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import normalize

from .model_trainer import ModelTrainer


def update_model_one_batch(optimizer, model, output, batch_target, loss_func, args):
    loss = loss_func(output, batch_target)
    optimizer.zero_grad()
    loss.backward()
    # 裁剪梯度,防止梯度爆炸
    # nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()
    return loss


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


def split_data(data, args):
    if args.dataset == 'Yahoo':
        x_b = data[1]
        x_a = data[0]
    elif args.dataset in ['CIFAR10', 'CIFAR100', 'CINIC10L']:
        x_a = data[:, :, :, 0:args.half]
        x_b = data[:, :, :, args.half:32]
    elif args.dataset == 'TinyImageNet':
        x_a = data[:, :, :, 0:args.half]
        x_b = data[:, :, :, args.half:64]
    elif args.dataset == 'Criteo':
        x_b = data[:, args.half:D_]
        x_a = data[:, 0:args.half]
    elif args.dataset == 'BCW':
        x_b = data[:, args.half:28]
        x_a = data[:, 0:args.half]
    else:
        raise Exception('Unknown dataset name!')
    return x_a, x_b


class VFLTrainer(ModelTrainer):
    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def update_model(self, new_model):
        # 更新self.model
        self.model = new_model

    def train(self, train_data, criterion, bottom_criterion, optimizer_list, device, args):

        model_list = self.model
        model_list = [model.to(device) for model in model_list]
        model_list = [model.train() for model in model_list]

        # train and update
        epoch_loss = []

        for step, (trn_X, trn_y) in enumerate(train_data):
            trn_X = [x.float().to(device) for x in trn_X]
            # target = torch.argmax(trn_y,dim=1).long().to(device)
            target = trn_y.float().to(device)
            batch_loss = []

            # input_tensor_top_model_a = torch.tensor([], requires_grad=True)
            # input_tensor_top_model_b = torch.tensor([], requires_grad=True)

            # bottom model B
            output_tensor_bottom_model_b = model_list[1](trn_X[1])
            # bottom model A
            output_tensor_bottom_model_a = model_list[0](trn_X[0])

            input_tensor_top_model_a = output_tensor_bottom_model_a.detach().clone()
            input_tensor_top_model_b = output_tensor_bottom_model_b.detach().clone()
            input_tensor_top_model_a.requires_grad_(True)
            input_tensor_top_model_b.requires_grad_(True)

            # top model
            output = model_list[2](input_tensor_top_model_a, input_tensor_top_model_b)
            # --top model backward/update--
            loss = update_model_one_batch(optimizer=optimizer_list[2],
                                          model=model_list[2],
                                          output=output,
                                          batch_target=target,
                                          loss_func=criterion,
                                          args=args)

            # read grad of: input of top model(also output of bottom models), which will be used as bottom model's target
            grad_output_bottom_model_a = input_tensor_top_model_a.grad
            grad_output_bottom_model_b = input_tensor_top_model_b.grad

            # -- bottom model b backward/update--
            _ = update_model_one_batch(optimizer=optimizer_list[1],
                                       model=model_list[1],
                                       output=output_tensor_bottom_model_b,
                                       batch_target=grad_output_bottom_model_b,
                                       loss_func=bottom_criterion,
                                       args=args)

            # -- bottom model a backward/update--
            _ = update_model_one_batch(optimizer=optimizer_list[0],
                                       model=model_list[0],
                                       output=output_tensor_bottom_model_a,
                                       batch_target=grad_output_bottom_model_a,
                                       loss_func=bottom_criterion,
                                       args=args)

            # logging.info('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #     epoch, (batch_idx + 1) * self.args.batch_size, len(self.local_training_data) * self.args.batch_size,
            #            100. * (batch_idx + 1) / len(self.local_training_data), loss.item()))
            batch_loss.append(loss.item())
        epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return epoch_loss

    def extract_features(self, train_data, device, args):

        victim_model = self.model[1].eval().to(device)

        features = []
        labels = []
        with torch.no_grad():
            for step, (trn_X, trn_y) in enumerate(train_data):
                if args.dataset in ['CIFAR10', 'CIFAR100', 'CINIC10L']:
                    trn_X = trn_X.float().to(device)
                    Xa, Xb = split_data(trn_X, args)
                    target = trn_y.long().to(device)
                else:
                    # trn_X = [x.float().to(device) for x in trn_X]
                    Xb = trn_X.float().to(device)
                    target = torch.argmax(trn_y, dim=1).long().to(device)
                # target = trn_y.float().to(device)
                batch_loss = []

                # bottom model B
                output_tensor_bottom_model_b = victim_model(Xb)
                features.append(output_tensor_bottom_model_b.detach().cpu().numpy())
                labels.append(target.detach().cpu().numpy())

        return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)

    def train_cluster(self, train_data, criterion, bottom_criterion, optimizer_list, device, args):

        # model_list = self.model

        # first three models adopt train(), last model adopts eval()
        model_list = [model.eval() for model in self.model[:-1]]
        model_list.append(self.model[-1].train())
        model_list = [model.to(device) for model in model_list]

        # train and update
        epoch_loss = []

        for step, (trn_X, trn_y) in enumerate(train_data):
            if args.dataset in ['CIFAR10', 'CIFAR100', 'CINIC10L']:
                trn_X = trn_X.float().to(device)
                Xa, Xb = split_data(trn_X, args)
                target = trn_y.long().to(device)
            else:
                # trn_X = [x.float().to(device) for x in trn_X]
                Xb = trn_X.float().to(device)
                # target = torch.argmax(trn_y, dim=1).long().to(device)
                target = trn_y.long().to(device)
            batch_loss = []

            # input_tensor_top_model_a = torch.tensor([], requires_grad=True)
            # input_tensor_top_model_b = torch.tensor([], requires_grad=True)

            # bottom model B
            output_tensor_bottom_model_b = model_list[1](Xb)
            # model completion head
            pseudo_output = model_list[3](output_tensor_bottom_model_b)

            input_tensor_top_model_b = output_tensor_bottom_model_b.detach().clone()
            input_tensor_top_model_b.requires_grad_(True)

            # --top model backward/update--
            loss = update_model_one_batch(optimizer=optimizer_list[0],
                                          model=model_list[3],
                                          output=pseudo_output,
                                          batch_target=target,
                                          loss_func=criterion,
                                          args=args)

            # read grad of: input of top model(also output of bottom models), which will be used as bottom model's target
            '''
            grad_output_bottom_model_b = input_tensor_top_model_b.grad

            # -- bottom model b backward/update--
            _ = update_model_one_batch(optimizer=optimizer_list[1],
                                       model=model_list[1],
                                       output=output_tensor_bottom_model_b,
                                       batch_target=grad_output_bottom_model_b,
                                       loss_func=bottom_criterion,
                                       args=args)
            '''

            # logging.info('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #     epoch, (batch_idx + 1) * self.args.batch_size, len(self.local_training_data) * self.args.batch_size,
            #            100. * (batch_idx + 1) / len(self.local_training_data), loss.item()))
            batch_loss.append(loss.item())
        epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return epoch_loss[0]

    def train_poisoning(self, train_data, criterion, bottom_criterion, optimizer_list, device, args, delta,
                        poisoned_indices):
        model_list = self.model
        model_list = [model.to(device) for model in model_list]
        model_list = [model.train() for model in model_list]

        # train and update
        epoch_loss = []
        poisoned_grads = []
        batch_poisoned_count = 0

        selected_set = set(poisoned_indices)

        # delta = torch.clamp(delta, -args.eps*2, args.eps*2)

        for step, (trn_X, trn_y, indices) in enumerate(train_data):

            if args.dataset in ['CIFAR10', 'CIFAR100', 'CINIC10L']:
                trn_X = trn_X.float().to(device)
                Xa, Xb = split_data(trn_X, args)
                target = trn_y.long().to(device)
            else:
                # trn_X = [x.float().to(device) for x in trn_X]
                Xa = trn_X[0].float().to(device)
                Xb = trn_X[1].float().to(device)
                target = trn_y.long().to(device)
            # target = trn_y.float().to(device)
            batch_loss = []

            '''
            selected_values = Xb[~mask]
            random_indices = torch.randint(0, (~mask).sum(), (mask.sum(),))
            Xb[mask] = selected_values[random_indices] + delta.detach().clone()
            #Xb[selected_values[random_indices]] = selected_values[random_indices] + delta.detach().clone()
            '''

            indices = indices.cpu().numpy()

            mask = torch.from_numpy(np.isin(indices, poisoned_indices))
            num_masked = mask.sum().item()

            # 假设Xb是你的数据张量，mask是你的掩码张量
            non_mask_indices = torch.where(~mask)[0]  # 找到非mask的索引
            random_indices = torch.randperm(non_mask_indices.size(0))[:num_masked]  # 从非mask的索引中随机选择5个

            Xb[non_mask_indices[random_indices]] += delta.detach().clone()  # 将delta添加到选定的Xb值

            # batch_delta.grad.zero_()
            # bottom model B
            output_tensor_bottom_model_b = model_list[1](Xb)
            # bottom model A
            output_tensor_bottom_model_a = model_list[0](Xa)

            input_tensor_top_model_a = output_tensor_bottom_model_a.detach().clone()
            input_tensor_top_model_b = output_tensor_bottom_model_b.detach().clone()

            # random-valued vector to replace the gradient of poisoned data
            random_grad = torch.randn(num_masked, input_tensor_top_model_b.shape[1]).to(device)
            input_tensor_top_model_b[non_mask_indices[random_indices]] = random_grad

            input_tensor_top_model_a.requires_grad_(True)
            input_tensor_top_model_b.requires_grad_(True)

            # top model
            output = model_list[2](input_tensor_top_model_a, input_tensor_top_model_b)
            # --top model backward/update--
            loss = update_model_one_batch(optimizer=optimizer_list[2],
                                          model=model_list[2],
                                          output=output,
                                          batch_target=target,
                                          loss_func=criterion,
                                          args=args)

            # read grad of: input of top model(also output of bottom models), which will be used as bottom model's target
            grad_output_bottom_model_a = input_tensor_top_model_a.grad
            grad_output_bottom_model_b = input_tensor_top_model_b.grad

            # Target Label replacement
            # 将grad_output_bottom_model_b中按mask中为True的元素索引赋值给non_mask_indices[random_indices]中的元素
            grad_output_bottom_model_b[non_mask_indices[random_indices]] = args.corruption_amp * \
                                                                           grad_output_bottom_model_b[mask]

            if args.max_norm:
                grad_output_bottom_model_b = gradient_masking(grad_output_bottom_model_b)
            # add iso gaussian noise
            if args.iso:
                grad_output_bottom_model_b = gradient_gaussian_noise_masking(grad_output_bottom_model_b,
                                                                             ratio=args.iso_ratio)
            # add marvell noise
            if args.marvell:
                grad_output_bottom_model_b = marvell_g(grad_output_bottom_model_b, target)

            # ----yjr-----
            # gradient compression
            if args.gc:
                grad_output_bottom_model_b = gradient_compression(grad_output_bottom_model_b,
                                                                  preserved_perc=args.gc_ratio)

            # differential privacy
            if args.lap_noise:
                grad_output_bottom_model_b = laplacian_noise_masking(grad_output_bottom_model_b,
                                                                     beta=args.lap_noise_ratio)

            # sign SGD
            # TODO: 这里的signSGD只是完成了在g上的操作，理论上应该使用optimizer.step()的方法来优化
            if args.signSGD:
                torch.sign(grad_output_bottom_model_b, out=grad_output_bottom_model_b)

            # -- bottom model b backward/update--
            _ = update_model_one_batch(optimizer=optimizer_list[1],
                                       model=model_list[1],
                                       output=output_tensor_bottom_model_b,
                                       batch_target=grad_output_bottom_model_b,
                                       loss_func=bottom_criterion,
                                       args=args)

            # -- bottom model a backward/update--
            _ = update_model_one_batch(optimizer=optimizer_list[0],
                                       model=model_list[0],
                                       output=output_tensor_bottom_model_a,
                                       batch_target=grad_output_bottom_model_a,
                                       loss_func=bottom_criterion,
                                       args=args)

            # poison_optimizer.step()

            # poisoned_grads.append(batch_delta_grad)

            # to avoid nan loss
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)

            # logging.info('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #     epoch, (batch_idx + 1) * self.args.batch_size, len(self.local_training_data) * self.args.batch_size,
            #            100. * (batch_idx + 1) / len(self.local_training_data), loss.item()))
            batch_loss.append(loss.item())

        epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return epoch_loss[0]

    def train_mul(self, train_data, criterion, bottom_criterion, optimizer_list, device, args):

        model_list = self.model
        model_list = [model.to(device) for model in model_list]
        model_list = [model.train() for model in model_list]

        # train and update
        epoch_loss = []

        for step, (trn_X, trn_y, indices) in enumerate(train_data):
            if args.dataset in ['CIFAR10', 'CIFAR100', 'CINIC10L']:
                trn_X = trn_X.float().to(device)
                Xa, Xb = split_data(trn_X, args)
                target = trn_y.long().to(device)
            else:
                # trn_X = [x.float().to(device) for x in trn_X]
                Xa = trn_X[0].float().to(device)
                Xb = trn_X[1].float().to(device)
                target = trn_y.long().to(device)
            # target = trn_y.float().to(device)
            batch_loss = []

            # input_tensor_top_model_a = torch.tensor([], requires_grad=True)
            # input_tensor_top_model_b = torch.tensor([], requires_grad=True)

            # bottom model B
            output_tensor_bottom_model_b = model_list[1](Xb)
            # bottom model A
            output_tensor_bottom_model_a = model_list[0](Xa)

            input_tensor_top_model_a = output_tensor_bottom_model_a.detach().clone()
            input_tensor_top_model_b = output_tensor_bottom_model_b.detach().clone()
            input_tensor_top_model_a.requires_grad_(True)
            input_tensor_top_model_b.requires_grad_(True)

            # top model
            output = model_list[2](input_tensor_top_model_a, input_tensor_top_model_b)
            # --top model backward/update--
            loss = update_model_one_batch(optimizer=optimizer_list[2],
                                          model=model_list[2],
                                          output=output,
                                          batch_target=target,
                                          loss_func=criterion,
                                          args=args)

            # read grad of: input of top model(also output of bottom models), which will be used as bottom model's target
            # TODO：！！！这里的grad用法很特别，需要学习！！！
            grad_output_bottom_model_a = input_tensor_top_model_a.grad
            grad_output_bottom_model_b = input_tensor_top_model_b.grad

            # -- bottom model b backward/update--
            _ = update_model_one_batch(optimizer=optimizer_list[1],
                                       model=model_list[1],
                                       output=output_tensor_bottom_model_b,
                                       batch_target=grad_output_bottom_model_b,
                                       loss_func=bottom_criterion,
                                       args=args)

            # -- bottom model a backward/update--
            _ = update_model_one_batch(optimizer=optimizer_list[0],
                                       model=model_list[0],
                                       output=output_tensor_bottom_model_a,
                                       batch_target=grad_output_bottom_model_a,
                                       loss_func=bottom_criterion,
                                       args=args)

            # logging.info('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #     epoch, (batch_idx + 1) * self.args.batch_size, len(self.local_training_data) * self.args.batch_size,
            #            100. * (batch_idx + 1) / len(self.local_training_data), loss.item()))
            batch_loss.append(loss.item())
        epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return epoch_loss[0]

    def test_cluster(self, test_data, criterion, device, args):
        model_list = self.model

        model_list = [model.to(device) for model in model_list]

        model_list = [model.eval() for model in model_list]

        test_loss = 0
        top5_correct = 0
        total = 0
        correct = 0

        with torch.no_grad():
            for batch_idx, (trn_X, trn_y) in enumerate(test_data):

                if args.dataset in ['CIFAR10', 'CIFAR100', 'CINIC10L']:
                    trn_X = trn_X.float().to(device)
                    Xa, Xb = split_data(trn_X, args)
                    target = trn_y.long().to(device)
                else:
                    # trn_X = [x.float().to(device) for x in trn_X]
                    Xb = trn_X.float().to(device)
                    target = torch.argmax(trn_y, dim=1).long().to(device)
                    # target = trn_y.long().to(device)

                # bottom model B
                output_tensor_bottom_model_b = model_list[1](Xb)

                # complete head
                output = model_list[-1](output_tensor_bottom_model_b)

                loss = criterion(output, target)
                test_loss += loss.item()  # sum up batch loss

                probs = F.softmax(output, dim=1)
                # Top-1 accuracy
                total += target.size(0)
                _, pred = probs.topk(1, 1, True, True)
                correct += torch.eq(pred, target.view(-1, 1)).sum().float().item()

                # Top-5 accuracy
                _, top5_preds = probs.topk(5, 1, True, True)
                top5_correct += torch.eq(top5_preds, target.view(-1, 1)).sum().float().item()

        test_loss = test_loss / total
        top1_acc = 100. * correct / total
        top5_acc = 100. * top5_correct / total

        print('Test set: Average loss: {:.4f}, Top-1 Accuracy: {}/{} ({:.2f}%), Top-5 Accuracy: {}/{} ({:.2f}%)'.format(
            test_loss, correct, total, top1_acc, top5_correct, total, top5_acc))

        return test_loss, top1_acc, top5_acc

    def test_copur_target(self, test_data, criterion, device, args, poison_target_label):
        model_list = self.model

        model_list = [model.to(device) for model in model_list]

        model_list = [model.eval() for model in model_list]

        test_loss = 0
        top5_correct = 0
        total = 0
        correct = 0
        asr_top5_correct = 0
        asr_correct = 0

        # batch_delta = delta.unsqueeze(0).repeat([poison_target_label.shape[0], 1, 1, 1]).detach().clone()

        for batch_idx, (trn_X, trn_y, indices) in enumerate(test_data):

            if args.dataset in ['CIFAR10', 'CIFAR100', 'CINIC10L']:
                trn_X = trn_X.float().to(device)
                Xa, Xb = split_data(trn_X, args)
                target = trn_y.long().to(device)
            else:
                # trn_X = [x.float().to(device) for x in trn_X]
                Xa = trn_X[0].float().to(device)
                Xb = trn_X[1].float().to(device)
                target = trn_y.long().to(device)

            target_class = torch.tensor([poison_target_label]).repeat(target.shape[0]).to(device)

            # bottom model B
            output_tensor_bottom_model_b = model_list[1](Xb)
            # bottom model A
            output_tensor_bottom_model_a = model_list[0](Xa)
            for i in range(50):
                # top model
                output = model_list[2](output_tensor_bottom_model_a, output_tensor_bottom_model_b)

                loss = criterion(output, target_class)

                # compute gradients of the loss with respect to output_tensor_bottom_model_b
                grad_output_bottom_model_b = torch.autograd.grad(loss, output_tensor_bottom_model_b)[0]

                adv_feat = output_tensor_bottom_model_b - 0.1 * grad_output_bottom_model_b.detach().clone()

                output_tensor_bottom_model_b = adv_feat.detach().clone()

            output = model_list[2](output_tensor_bottom_model_a, output_tensor_bottom_model_b)

            probs = F.softmax(output, dim=1)
            # Top-1 accuracy
            total += target.size(0)
            _, pred = probs.topk(1, 1, True, True)
            _, top5_preds = probs.topk(5, 1, True, True)

            asr_correct += torch.eq(pred, target_class.view(-1, 1)).sum().float().item()

            # Top-5 accuracy

            asr_top5_correct += torch.eq(top5_preds, target_class.view(-1, 1)).sum().float().item()

        asr_top1_acc = 100. * asr_correct / total
        asr_top5_acc = 100. * asr_top5_correct / total

        print(
            'CoPur Test set: ASR Top-1 Accuracy: {}/{} ({:.4f}%), ASR Top-5 Accuracy: {}/{} ({:.4f}%)'
            .format(
                asr_correct, total, asr_top1_acc, asr_top5_correct, total, asr_top5_acc))

        return asr_top1_acc, asr_top5_acc

    def test_mul(self, test_data, criterion, device, args):
        model_list = self.model

        model_list = [model.to(device) for model in model_list]

        model_list = [model.eval() for model in model_list]

        test_loss = 0
        top5_correct = 0
        total = 0
        correct = 0

        with torch.no_grad():
            for batch_idx, (trn_X, trn_y, indices) in enumerate(test_data):

                if args.dataset in ['CIFAR10', 'CIFAR100', 'CINIC10L']:
                    trn_X = trn_X.float().to(device)
                    Xa, Xb = split_data(trn_X, args)
                    target = trn_y.long().to(device)
                else:
                    # trn_X = [x.float().to(device) for x in trn_X]
                    Xa = trn_X[0].float().to(device)
                    Xb = trn_X[1].float().to(device)
                    target = trn_y.long().to(device)

                # bottom model B
                output_tensor_bottom_model_b = model_list[1](Xb)
                # bottom model A
                output_tensor_bottom_model_a = model_list[0](Xa)

                # top model
                output = model_list[2](output_tensor_bottom_model_a, output_tensor_bottom_model_b)

                loss = criterion(output, target)
                test_loss += loss.item()  # sum up batch loss

                probs = F.softmax(output, dim=1)
                # Top-1 accuracy
                total += target.size(0)
                _, pred = probs.topk(1, 1, True, True)
                correct += torch.eq(pred, target.view(-1, 1)).sum().float().item()

                # Top-5 accuracy
                _, top5_preds = probs.topk(5, 1, True, True)
                top5_correct += torch.eq(top5_preds, target.view(-1, 1)).sum().float().item()

        test_loss = test_loss / total
        top1_acc = 100. * correct / total
        top5_acc = 100. * top5_correct / total

        print('Test set: Average loss: {:.4f}, Top-1 Accuracy: {}/{} ({:.4f}%), Top-5 Accuracy: {}/{} ({:.4f}%)'.format(
            test_loss, correct, total, top1_acc, top5_correct, total, top5_acc))

        return test_loss, top1_acc, top5_acc

    def test_backdoor_mul(self, test_data, criterion, device, args, delta, poison_target_label):
        model_list = self.model

        model_list = [model.to(device) for model in model_list]

        model_list = [model.eval() for model in model_list]

        test_loss = 0
        top5_correct = 0
        total = 0
        correct = 0

        # batch_delta = delta.unsqueeze(0).repeat([poison_target_label.shape[0], 1, 1, 1]).detach().clone()

        with torch.no_grad():
            for batch_idx, (trn_X, trn_y, indices) in enumerate(test_data):

                if args.dataset in ['CIFAR10', 'CIFAR100', 'CINIC10L']:
                    trn_X = trn_X.float().to(device)
                    Xa, Xb = split_data(trn_X, args)
                    target = trn_y.long().to(device)
                else:
                    # trn_X = [x.float().to(device) for x in trn_X]
                    Xa = trn_X[0].float().to(device)
                    Xb = trn_X[1].float().to(device)
                    target = trn_y.long().to(device)

                target_class = torch.tensor([poison_target_label]).repeat(target.shape[0]).to(device)

                # bottom model B
                output_tensor_bottom_model_b = model_list[1](Xb + delta)
                # bottom model A
                output_tensor_bottom_model_a = model_list[0](Xa)

                # top model
                output = model_list[2](output_tensor_bottom_model_a, output_tensor_bottom_model_b)

                loss = criterion(output, target)
                test_loss += loss.item()  # sum up batch loss

                probs = F.softmax(output, dim=1)
                # Top-1 accuracy
                total += target.size(0)
                _, pred = probs.topk(1, 1, True, True)
                correct += torch.eq(pred, target_class.view(-1, 1)).sum().float().item()

                # Top-5 accuracy
                _, top5_preds = probs.topk(5, 1, True, True)
                top5_correct += torch.eq(top5_preds, target_class.view(-1, 1)).sum().float().item()

        test_loss = test_loss / total
        top1_acc = 100. * correct / total
        top5_acc = 100. * top5_correct / total

        print(
            'Backdoor Test set: Average loss: {:.4f}, ASR Top-1 Accuracy: {}/{} ({:.4f}%, ASR Top-5 Accuracy: {}/{} ({:.4f}%)'.format(
                test_loss, correct, total, top1_acc, top5_correct, total, top5_acc))

        return test_loss, top1_acc, top5_acc

    def check_backdoor(self, test_data, criterion, device, args, poison_target_label, delta=None, cut_ratio=[1.,1.]):
        model_list = self.model

        model_list = [model.to(device) for model in model_list]

        model_list = [model.eval() for model in model_list]

        test_loss = 0
        top5_correct = 0
        total = 0
        correct = 0

        # batch_delta = delta.unsqueeze(0).repeat([poison_target_label.shape[0], 1, 1, 1]).detach().clone()

        with torch.no_grad():
            for batch_idx, (trn_X, trn_y, indices) in enumerate(test_data):

                if args.dataset in ['CIFAR10', 'CIFAR100', 'CINIC10L']:
                    trn_X = trn_X.float().to(device)
                    Xa, Xb = split_data(trn_X, args)
                    target = trn_y.long().to(device)
                else:
                    # trn_X = [x.float().to(device) for x in trn_X]
                    Xa = trn_X[0].float().to(device)
                    Xb = trn_X[1].float().to(device)
                    target = trn_y.long().to(device)

                target_class = torch.tensor([poison_target_label]).repeat(target.shape[0]).to(device)

                # bottom model B
                if delta is not None:
                    output_tensor_bottom_model_b = model_list[1](Xb + delta)
                else:
                    output_tensor_bottom_model_b = model_list[1](Xb)
                # bottom model A
                output_tensor_bottom_model_a = model_list[0](Xa)

                # top model
                output = model_list[2](output_tensor_bottom_model_a*cut_ratio[0], output_tensor_bottom_model_b*cut_ratio[1])

                loss = criterion(output, target)
                test_loss += loss.item()  # sum up batch loss

                probs = F.softmax(output, dim=1)
                # Top-1 accuracy
                total += target.size(0)
                _, pred = probs.topk(1, 1, True, True)
                correct += torch.eq(pred, target_class.view(-1, 1)).sum().float().item()

                # Top-5 accuracy
                _, top5_preds = probs.topk(5, 1, True, True)
                top5_correct += torch.eq(top5_preds, target_class.view(-1, 1)).sum().float().item()

        test_loss = test_loss / total
        top1_acc = 100. * correct / total
        top5_acc = 100. * top5_correct / total

        # print(
        #     'Backdoor Test set: Average loss: {:.4f}, ASR Top-1 Accuracy: {}/{} ({:.4f}%, ASR Top-5 Accuracy: {}/{} ({:.4f}%)'.format(
        #         test_loss, correct, total, top1_acc, top5_correct, total, top5_acc))

        return test_loss, top1_acc, top5_acc

    def test(self, test_data, criterion, device):
        model_list = self.model

        model_list = [model.to(device) for model in model_list]

        model_list = [model.eval() for model in model_list]

        m = nn.Sigmoid()

        Loss = AverageMeter()
        AUC = AverageMeter()
        ACC = AverageMeter()
        Precision = AverageMeter()
        Recall = AverageMeter()
        F1 = AverageMeter()
        ASR_0 = AverageMeter()
        ASR_1 = AverageMeter()
        ARR_0 = AverageMeter()
        ARR_1 = AverageMeter()

        '''
        stackoverflow_lr is the task of multi-label classification
        please refer to following links for detailed explainations on cross-entropy and corresponding implementation of tff research:
        https://towardsdatascience.com/cross-entropy-for-classification-d98e7f974451
        https://github.com/google-research/federated/blob/49a43456aa5eaee3e1749855eed89c0087983541/optimization/stackoverflow_lr/federated_stackoverflow_lr.py#L131
        '''
        with torch.no_grad():
            for batch_idx, (trn_X, target) in enumerate(test_data):
                trn_X = [x.float().to(device) for x in trn_X]
                target = target.float().to(device)

                # bottom model B
                output_tensor_bottom_model_b = model_list[1](trn_X[1])
                # bottom model A
                output_tensor_bottom_model_a = model_list[0](trn_X[0])

                # top model
                pred = m(model_list[2](output_tensor_bottom_model_a, output_tensor_bottom_model_b))

                loss = criterion(pred, target)

                y_hat_lbls, statistics = compute_correct_prediction(y_targets=target,
                                                                    y_prob_preds=pred,
                                                                    threshold=0.5)

                acc = accuracy_score(target.cpu().numpy(), y_hat_lbls)
                # auc = roc_auc_score(target.cpu().numpy(), pred.cpu().numpy())

                try:
                    auc = roc_auc_score(target.cpu().numpy(), pred.cpu().numpy())
                except  ValueError:
                    pass

                metrics = precision_recall_fscore_support(target.cpu().numpy(), y_hat_lbls, average="binary",
                                                          warn_for=tuple())
                # print(classification_report(target.cpu().numpy(), y_hat_lbls))
                cm = confusion_matrix(target.cpu().numpy(), y_hat_lbls)

                poisoned_fn_rate, poisoned_tn_rate, poisoned_fp_rate, poisoned_tp_rate = backdoor_truepostive_rate(cm)
                '''
                predicted = (pred > .5).int()
                correct = predicted.eq(target).sum(axis=-1).eq(target.size(1)).sum()
                true_positive = ((target * predicted) > .1).int().sum(axis=-1)
                precision = true_positive / (predicted.sum(axis=-1) + 1e-13)
                recall = true_positive / (target.sum(axis=-1) + 1e-13)
                metrics['test_precision'] += precision.sum().item()
                metrics['test_recall'] += recall.sum().item()
                metrics['test_correct'] += correct.item()
                metrics['test_loss'] += loss.item() * target.size(0)
                metrics['test_total'] += target.size(0)
                '''
                ACC.update(acc)
                AUC.update(auc)
                Loss.update(loss)
                Precision.update(metrics[0])
                Recall.update(metrics[1])
                F1.update(metrics[2])
                ASR_0.update(poisoned_fn_rate)
                ASR_1.update(poisoned_fp_rate)
                ARR_0.update(poisoned_tn_rate)
                ARR_1.update(poisoned_tp_rate)

        return Loss.avg, ACC.avg, AUC.avg, Precision.avg, Recall.avg, F1.avg, ASR_0.avg, ASR_1.avg, ARR_0.avg, ARR_1.avg

    def train_narcissus(self, train_data, criterion, bottom_criterion, optimizer_list, device, args, delta,
                        poisoned_indices):
        # poisoned_indices: 是一个要毒害的index的list
        model_list = self.model
        model_list = [model.to(device) for model in model_list]
        model_list = [model.train() for model in model_list]

        # train and update
        epoch_loss = []
        poisoned_grads = []
        batch_poisoned_count = 0

        selected_set = set(poisoned_indices)

        # delta = torch.clamp(delta, -args.eps*2, args.eps*2)

        for step, (trn_X, trn_y, indices) in enumerate(train_data):

            if args.dataset in ['CIFAR10', 'CIFAR100', 'CINIC10L']:
                trn_X = trn_X.float().to(device)
                Xa, Xb = split_data(trn_X, args)
                target = trn_y.long().to(device)
            else:
                # trn_X = [x.float().to(device) for x in trn_X]
                Xa = trn_X[0].float().to(device)
                Xb = trn_X[1].float().to(device)
                target = trn_y.long().to(device)
            # target = trn_y.float().to(device)
            batch_loss = []

            indices = indices.cpu().numpy()

            batch_delta = torch.zeros_like(Xb).to(device)
            # mask标记哪些是要毒害的
            mask = torch.from_numpy(np.isin(indices, poisoned_indices))
            batch_delta[mask] = delta.detach().clone()
            batch_delta.requires_grad_()

            # batch_delta.grad.zero_()
            # bottom model B
            # 直接在上面加上随机扰动？
            output_tensor_bottom_model_b = model_list[1](Xb + batch_delta)
            # bottom model A
            output_tensor_bottom_model_a = model_list[0](Xa)

            # corruption_amp
            # output_tensor_bottom_model_a[mask]*=args.corruption_amp

            input_tensor_top_model_a = output_tensor_bottom_model_a.detach().clone()
            input_tensor_top_model_b = output_tensor_bottom_model_b.detach().clone()
            input_tensor_top_model_a.requires_grad_(True)
            input_tensor_top_model_b.requires_grad_(True)

            # top model
            output = model_list[2](input_tensor_top_model_a, input_tensor_top_model_b)
            # --top model backward/update--
            loss = update_model_one_batch(optimizer=optimizer_list[2],
                                          model=model_list[2],
                                          output=output,
                                          batch_target=target,
                                          loss_func=criterion,
                                          args=args)

            # read grad of: input of top model(also output of bottom models), which will be used as bottom model's target
            grad_output_bottom_model_a = input_tensor_top_model_a.grad
            grad_output_bottom_model_b = input_tensor_top_model_b.grad

            if args.max_norm:
                grad_output_bottom_model_b = gradient_masking(grad_output_bottom_model_b)
            # add iso gaussian noise
            if args.iso:
                grad_output_bottom_model_b = gradient_gaussian_noise_masking(grad_output_bottom_model_b,
                                                                             ratio=args.iso_ratio)
            # add marvell noise
            if args.marvell:
                grad_output_bottom_model_b = marvell_g(grad_output_bottom_model_b, target)
            # ----yjr-----
            # gradient compression
            if args.gc:
                grad_output_bottom_model_b = gradient_compression(grad_output_bottom_model_b,
                                                                  preserved_perc=args.gc_ratio)

            # differential privacy
            if args.lap_noise:
                grad_output_bottom_model_b = laplacian_noise_masking(grad_output_bottom_model_b,
                                                                     beta=args.lap_noise_ratio)

            # sign SGD
            # TODO: 这里的signSGD只是完成了在g上的操作，理论上应该使用optimizer.step()的方法来优化
            if args.signSGD:
                torch.sign(grad_output_bottom_model_b, out=grad_output_bottom_model_b)

            # -- bottom model b backward/update--
            _ = update_model_one_batch(optimizer=optimizer_list[1],
                                       model=model_list[1],
                                       output=output_tensor_bottom_model_b,
                                       batch_target=grad_output_bottom_model_b,
                                       loss_func=bottom_criterion,
                                       args=args)

            # -- bottom model a backward/update--
            _ = update_model_one_batch(optimizer=optimizer_list[0],
                                       model=model_list[0],
                                       output=output_tensor_bottom_model_a,
                                       batch_target=grad_output_bottom_model_a,
                                       loss_func=bottom_criterion,
                                       args=args)

            # norm_attack_

            # Calculate the squared sum of the gradients along dim=1
            # squared_sum = grad_output_bottom_model_b.pow(2).sum(dim=1)

            # Ensure that the squared sum has no negative values
            # non_negative_squared_sum = torch.clamp(squared_sum, min=0)

            # Take the square root of the non-negative squared sum
            # g_norm = torch.sqrt(non_negative_squared_sum)

            # UAB attack
            batch_delta_grad = batch_delta.grad[mask]
            poisoned_grads.append(batch_delta_grad)

            # poisoned_grads = torch.cat(poisoned_grads, dim=0)

            # poison_optimizer.zero_grad()

            grad_sign = batch_delta_grad.detach().mean(dim=0).sign()

            delta = delta - grad_sign * args.alpha

            # delta.grad = poisoned_avg_grads.unsqueeze(0)
            # delta.backward(poisoned_avg_grads)

            delta = torch.clamp(delta, -args.eps, args.eps)

            # poison_optimizer.step()

            # poisoned_grads.append(batch_delta_grad)
            # l_inf

            '''
            # l2
            grad_norms = torch.norm(adv_gradients[0][0].view(n, -1), p=2, dim=1) + 1e-10
            grad = adv_gradients[0][0] / grad_norms.view(n, 1)
            grad_sign = grad.data.mean(dim=0).sign()
            delta = delta + grad_sign * args.alpha

            delta_norms = torch.norm(delta.view(trn_X[1].shape[1], -1), p=2, dim=1)
            factor = args.eps/ delta_norms
            factor = torch.min(factor, torch.ones_like(delta_norms))
            delta = delta * factor
            '''

            '''
            grad_norms = torch.norm(adv_gradients[0][0].view(n, -1), p=2, dim=1) + 1e-10
            grad = adv_gradients[0][0] / grad_norms.view(n, 1)
            grad = grad + momentum * args.decay
            momentum = grad.detach()

            grad_sign = grad.data.mean(dim=0).sign()
            delta = delta + grad_sign * args.alpha
            '''

            # to avoid nan loss
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)

            # logging.info('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #     epoch, (batch_idx + 1) * self.args.batch_size, len(self.local_training_data) * self.args.batch_size,
            #            100. * (batch_idx + 1) / len(self.local_training_data), loss.item()))
            batch_loss.append(loss.item())

        '''
        poisoned_grads = torch.cat(poisoned_grads, dim=0)
        
        #poisoned_avg_grads = poisoned_grads.detach().mean(dim=0)

        #delta.grad = poisoned_avg_grads.unsqueeze(0)
        #delta.backward(poisoned_avg_grads.unsqueeze(0))

        #poison_optimizer.step()
        
        grad_sign = poisoned_grads.detach().mean(dim=0).sign()
        delta = delta - grad_sign * args.alpha
        # revise hyper-parameters self.alpha, self.eps and final adv_Xb clamp
        delta = torch.clamp(delta, -args.eps, args.eps)
        '''

        epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return epoch_loss[0], delta

    def train_backdoor(self, train_data, criterion, bottom_criterion, optimizer_list, device, args, delta,
                       poisoned_indices):

        model_list = self.model
        model_list = [model.to(device) for model in model_list]
        model_list = [model.train() for model in model_list]

        # train and update
        epoch_loss = []

        for step, (trn_X, trn_y, indices) in enumerate(train_data):

            if args.dataset in ['CIFAR10', 'CIFAR100', 'CINIC10L']:
                trn_X = trn_X.float().to(device)
                Xa, Xb = split_data(trn_X, args)
                target = trn_y.long().to(device)
            else:
                # trn_X = [x.float().to(device) for x in trn_X]
                Xa = trn_X[0].float().to(device)
                Xb = trn_X[1].float().to(device)
                target = torch.argmax(trn_y, dim=1).long().to(device)
            # target = trn_y.float().to(device)
            batch_loss = []

            batch_delta = delta.unsqueeze(0).repeat([target.shape[0], 1]).detach().clone()
            batch_delta.requires_grad_()

            # batch_delta.grad.zero_()
            # bottom model B
            # 这里是在每一个batch里面，对每一个样本进行都进行+delta的操作
            output_tensor_bottom_model_b = model_list[1](Xb + batch_delta)
            # bottom model A
            output_tensor_bottom_model_a = model_list[0](Xa)

            input_tensor_top_model_a = output_tensor_bottom_model_a.detach().clone()
            input_tensor_top_model_b = output_tensor_bottom_model_b.detach().clone()
            input_tensor_top_model_a.requires_grad_(True)
            input_tensor_top_model_b.requires_grad_(True)

            # top model
            output = model_list[2](input_tensor_top_model_a, input_tensor_top_model_b)
            # --top model backward/update--
            loss = update_model_one_batch(optimizer=optimizer_list[2],
                                          model=model_list[2],
                                          output=output,
                                          batch_target=target,
                                          loss_func=criterion,
                                          args=args)

            # read grad of: input of top model(also output of bottom models), which will be used as bottom model's target
            grad_output_bottom_model_a = input_tensor_top_model_a.grad
            grad_output_bottom_model_b = input_tensor_top_model_b.grad

            # print(grad_output_bottom_model_a.size())
            # print(grad_output_bottom_model_b.size())
            # 下面内容貌似没有用到
            if args.max_norm:
                grad_output_bottom_model_b = gradient_masking(grad_output_bottom_model_b)
            # add iso gaussian noise
            if args.iso:
                grad_output_bottom_model_b = gradient_gaussian_noise_masking(grad_output_bottom_model_b,
                                                                             ratio=args.iso_ratio)
            # add marvell noise
            if args.marvell:
                grad_output_bottom_model_b = marvell_g(grad_output_bottom_model_b, target)
                # U_B_gradients_list = marvell_g(U_B_gradients_list, target)
            # ----yjr-----
            # gradient compression
            if args.gc:
                U_B_gradients_list = gradient_compression(U_B_gradients_list, preserved_perc=args.gc_ratio)

            # differential privacy
            if args.lap_noise:
                U_B_gradients_list = laplacian_noise_masking(U_B_gradients_list, beta=args.lap_noise_ratio)

            # sign SGD
            # TODO: 这里的signSGD只是完成了在g上的操作，理论上应该使用optimizer.step()的方法来优化
            if args.signSGD:
                torch.sign(U_B_gradients_list, out=U_B_gradients_list)

            # -- bottom model b backward/update--
            _ = update_model_one_batch(optimizer=optimizer_list[1],
                                       model=model_list[1],
                                       output=output_tensor_bottom_model_b,
                                       batch_target=grad_output_bottom_model_b,
                                       loss_func=bottom_criterion,
                                       args=args)

            # -- bottom model a backward/update--
            _ = update_model_one_batch(optimizer=optimizer_list[0],
                                       model=model_list[0],
                                       output=output_tensor_bottom_model_a,
                                       batch_target=grad_output_bottom_model_a,
                                       loss_func=bottom_criterion,
                                       args=args)

            # norm_attack_
            g_norm = grad_output_bottom_model_b.pow(2).sum(dim=1).sqrt()
            # Calculate the squared sum of the gradients along dim=1
            # squared_sum = grad_output_bottom_model_b.pow(2).sum(dim=1)

            # Ensure that the squared sum has no negative values
            # non_negative_squared_sum = torch.clamp(squared_sum, min=0)

            # Take the square root of the non-negative squared sum
            # g_norm = torch.sqrt(non_negative_squared_sum)

            score = roc_auc_score(target.cpu().numpy(), g_norm.cpu().numpy())

            print("norm attack score: ", score)
            # target =0
            if args.negative_target:
                topk_values, topk_indices = torch.topk(g_norm.view(-1), k=args.Topk)
            else:
                # target =1
                topk_values, topk_indices = torch.topk(g_norm.view(-1), k=args.Topk, largest=False)

            # UAB attack
            batch_delta_grad = batch_delta.grad
            # l_inf
            grad_sign = batch_delta_grad[topk_indices].data.mean(dim=0).sign()
            delta = delta + grad_sign * args.alpha
            # revise hyper-parameters self.alpha, self.eps and final adv_Xb clamp
            delta = torch.clamp(delta, -args.eps, args.eps)
            '''
            # l2
            grad_norms = torch.norm(adv_gradients[0][0].view(n, -1), p=2, dim=1) + 1e-10
            grad = adv_gradients[0][0] / grad_norms.view(n, 1)
            grad_sign = grad.data.mean(dim=0).sign()
            delta = delta + grad_sign * args.alpha

            delta_norms = torch.norm(delta.view(trn_X[1].shape[1], -1), p=2, dim=1)
            factor = args.eps/ delta_norms
            factor = torch.min(factor, torch.ones_like(delta_norms))
            delta = delta * factor
            '''

            '''
            grad_norms = torch.norm(adv_gradients[0][0].view(n, -1), p=2, dim=1) + 1e-10
            grad = adv_gradients[0][0] / grad_norms.view(n, 1)
            grad = grad + momentum * args.decay
            momentum = grad.detach()

            grad_sign = grad.data.mean(dim=0).sign()
            delta = delta + grad_sign * args.alpha
            '''

            # to avoid nan loss
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)

            # logging.info('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #     epoch, (batch_idx + 1) * self.args.batch_size, len(self.local_training_data) * self.args.batch_size,
            #            100. * (batch_idx + 1) / len(self.local_training_data), loss.item()))
            batch_loss.append(loss.item())
        epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return epoch_loss, delta

    def test_backdoor(self, test_data, criterion, device, delta):
        model_list = self.model

        model_list = [model.to(device) for model in model_list]

        model_list = [model.eval() for model in model_list]

        m = nn.Sigmoid()

        Loss = AverageMeter()
        AUC = AverageMeter()
        ACC = AverageMeter()
        Precision = AverageMeter()
        Recall = AverageMeter()
        F1 = AverageMeter()
        ASR_0 = AverageMeter()
        ASR_1 = AverageMeter()
        ARR_0 = AverageMeter()
        ARR_1 = AverageMeter()

        '''
        stackoverflow_lr is the task of multi-label classification
        please refer to following links for detailed explainations on cross-entropy and corresponding implementation of tff research:
        https://towardsdatascience.com/cross-entropy-for-classification-d98e7f974451
        https://github.com/google-research/federated/blob/49a43456aa5eaee3e1749855eed89c0087983541/optimization/stackoverflow_lr/federated_stackoverflow_lr.py#L131
        '''
        with torch.no_grad():
            for batch_idx, (trn_X, target) in enumerate(test_data):
                trn_X = [x.float().to(device) for x in trn_X]
                target = target.float().to(device)

                # bottom model B
                output_tensor_bottom_model_b = model_list[1](trn_X[1] + delta)
                # bottom model A
                output_tensor_bottom_model_a = model_list[0](trn_X[0])

                # top model
                pred = m(model_list[2](output_tensor_bottom_model_a, output_tensor_bottom_model_b))

                loss = criterion(pred, target)

                y_hat_lbls, statistics = compute_correct_prediction(y_targets=target,
                                                                    y_prob_preds=pred,
                                                                    threshold=0.5)

                acc = accuracy_score(target.cpu().numpy(), y_hat_lbls)
                # auc = roc_auc_score(target.cpu().numpy(), pred.cpu().numpy())

                try:
                    auc = roc_auc_score(target.cpu().numpy(), pred.cpu().numpy())
                except  ValueError:
                    pass

                metrics = precision_recall_fscore_support(target.cpu().numpy(), y_hat_lbls, average="binary",
                                                          warn_for=tuple())
                # print(classification_report(target.cpu().numpy(), y_hat_lbls))
                cm = confusion_matrix(target.cpu().numpy(), y_hat_lbls)

                poisoned_fn_rate, poisoned_tn_rate, poisoned_fp_rate, poisoned_tp_rate = backdoor_truepostive_rate(cm)
                '''
                predicted = (pred > .5).int()
                correct = predicted.eq(target).sum(axis=-1).eq(target.size(1)).sum()
                true_positive = ((target * predicted) > .1).int().sum(axis=-1)
                precision = true_positive / (predicted.sum(axis=-1) + 1e-13)
                recall = true_positive / (target.sum(axis=-1) + 1e-13)
                metrics['test_precision'] += precision.sum().item()
                metrics['test_recall'] += recall.sum().item()
                metrics['test_correct'] += correct.item()
                metrics['test_loss'] += loss.item() * target.size(0)
                metrics['test_total'] += target.size(0)
                '''
                ACC.update(acc)
                AUC.update(auc)
                Loss.update(loss)
                Precision.update(metrics[0])
                Recall.update(metrics[1])
                F1.update(metrics[2])
                ASR_0.update(poisoned_fn_rate)
                ASR_1.update(poisoned_fp_rate)
                ARR_0.update(poisoned_tn_rate)
                ARR_1.update(poisoned_tp_rate)

        return Loss.avg, ACC.avg, AUC.avg, Precision.avg, Recall.avg, F1.avg, ASR_0.avg, ASR_1.avg, ARR_0.avg, ARR_1.avg

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False
