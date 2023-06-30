import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init

import torch.nn.functional as F


def weights_init(m):
    # classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, kernel_size, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=kernel_size, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant",
                                                  0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, kernel_size, num_classes):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=kernel_size, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], kernel_size, stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], kernel_size, stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], kernel_size, stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(weights_init)

    def _make_layer(self, block, planes, num_blocks, kernel_size, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, kernel_size, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        # [bs,3,32,16]
        out = F.relu(self.bn1(self.conv1(x)))
        # [bs,16,32,16]
        out = self.layer1(out)
        # [bs,16,32,16]
        out = self.layer2(out)
        # [bs,32,16,8]
        out = self.layer3(out)
        # [bs,64,8,4]
        out = F.avg_pool2d(out, out.size()[2:])
        # [bs,64,1,1]
        out = out.view(out.size(0), -1)
        # [bs,64]
        out = self.linear(out)
        # [bs,10]
        return out


def resnet20(kernel_size=(3, 3), num_classes=10):
    return ResNet(block=BasicBlock, num_blocks=[3, 3, 3], kernel_size=kernel_size, num_classes=num_classes)


class BottomModelForCifar100(nn.Module):
    def __init__(self):
        super(BottomModelForCifar100, self).__init__()
        self.resnet20 = resnet20(num_classes=100)

    def forward(self, x):
        x = self.resnet20(x)
        return x


class TopModelForCifar100(nn.Module):
    def __init__(self):
        super(TopModelForCifar100, self).__init__()
        self.fc1top = nn.Linear(200, 200)
        self.fc2top = nn.Linear(200, 100)
        self.fc3top = nn.Linear(100, 100)
        self.fc4top = nn.Linear(100, 100)
        self.bn0top = nn.BatchNorm1d(200)
        self.bn1top = nn.BatchNorm1d(200)
        self.bn2top = nn.BatchNorm1d(100)
        self.bn3top = nn.BatchNorm1d(100)

        self.apply(weights_init)

    def forward(self, input_tensor_top_model_a, input_tensor_top_model_b):
        output_bottom_models = torch.cat((input_tensor_top_model_a, input_tensor_top_model_b), dim=1)
        x = output_bottom_models
        x = self.fc1top(F.relu(self.bn0top(x)))
        x = self.fc2top(F.relu(self.bn1top(x)))
        x = self.fc3top(F.relu(self.bn2top(x)))
        x = self.fc4top(F.relu(self.bn3top(x)))
        return F.log_softmax(x, dim=1)



class BottomModelForCinic10(nn.Module):
    def __init__(self):
        super(BottomModelForCinic10, self).__init__()
        self.resnet20 = resnet20(num_classes=10)

    def forward(self, x):
        x = self.resnet20(x)
        return x


class TopModelForCinic10(nn.Module):
    def __init__(self):
        super(TopModelForCinic10, self).__init__()
        self.fc1top = nn.Linear(20, 20)
        self.fc2top = nn.Linear(20, 10)
        self.fc3top = nn.Linear(10, 10)
        self.fc4top = nn.Linear(10, 10)
        self.bn0top = nn.BatchNorm1d(20)
        self.bn1top = nn.BatchNorm1d(20)
        self.bn2top = nn.BatchNorm1d(10)
        self.bn3top = nn.BatchNorm1d(10)

        self.apply(weights_init)

    def forward(self, input_tensor_top_model_a, input_tensor_top_model_b):
        output_bottom_models = torch.cat((input_tensor_top_model_a, input_tensor_top_model_b), dim=1)
        x = output_bottom_models
        x = self.fc1top(F.relu(self.bn0top(x)))
        x = self.fc2top(F.relu(self.bn1top(x)))
        x = self.fc3top(F.relu(self.bn2top(x)))
        x = self.fc4top(F.relu(self.bn3top(x)))
        return F.log_softmax(x, dim=1)


class BottomModelForCifar10(nn.Module):
    def __init__(self):
        super(BottomModelForCifar10, self).__init__()
        self.resnet20 = resnet20(num_classes=10)

    def forward(self, x):
        x = self.resnet20(x)
        return x


class TopModelForCifar10(nn.Module):
    def __init__(self):
        super(TopModelForCifar10, self).__init__()
        self.fc1top = nn.Linear(20, 20)
        self.fc2top = nn.Linear(20, 10)
        self.fc3top = nn.Linear(10, 10)
        self.fc4top = nn.Linear(10, 10)
        self.bn0top = nn.BatchNorm1d(20)
        self.bn1top = nn.BatchNorm1d(20)
        self.bn2top = nn.BatchNorm1d(10)
        self.bn3top = nn.BatchNorm1d(10)

        self.apply(weights_init)

    def forward(self, input_tensor_top_model_a, input_tensor_top_model_b):
        output_bottom_models = torch.cat((input_tensor_top_model_a, input_tensor_top_model_b), dim=1)
        x = output_bottom_models
        x = self.fc1top(F.relu(self.bn0top(x)))
        x = self.fc2top(F.relu(self.bn1top(x)))
        x = self.fc3top(F.relu(self.bn2top(x)))
        x = self.fc4top(F.relu(self.bn3top(x)))
        return F.log_softmax(x, dim=1)




'''
class active_model(nn.Module):
    def __init__(self, input_dim, num_classes, k=2):
        super(active_model, self).__init__()
        self.classifier = nn.Linear(in_features=input_dim * k, out_features=num_classes)
    def forward(self, input, U_B):
        #out = torch.cat([input] + [U for U in U_B], dim=1)
        out = torch.cat((input, U_B), dim=1)
        out = F.leaky_relu(out)
        logits = self.classifier(out)
        return logits

class ActiveModelWithOneLayer(nn.Module):
    def __init__(self, input_dim, num_classes, k=2):
        super(ActiveModelWithOneLayer, self).__init__()
        self.classifier = nn.Linear(in_features=input_dim * k, out_features=num_classes)
    def forward(self, input):
        #out = F.leaky_relu(out)
        logits = self.classifier(input)
        return logits

class passive_model(nn.Module):
    def __init__(self, input_dim, intern_dim, output_dim):
        super(passive_model, self).__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, intern_dim),
            nn.LeakyReLU(),
            nn.Linear(intern_dim, intern_dim),
            nn.LeakyReLU(),
            nn.Linear(in_features=intern_dim, out_features=output_dim)
        )
        
        self.mlp = nn.Sequential(nn.Linear(input_dim, intern_dim),
            nn.LeakyReLU(),
            nn.Linear(in_features=intern_dim, out_features=output_dim))

    def forward(self, input):
        out = self.mlp(input)
        return out
'''

class active_model(nn.Module):
    def __init__(self, input_dim, num_classes, k=2):
        super(active_model, self).__init__()
        self.classifier = nn.Linear(in_features=input_dim * k, out_features=num_classes)
    def forward(self, input, U_B):
        #out = torch.cat([input] + [U for U in U_B], dim=1)
        out = torch.cat((input, U_B), dim=1)
        #out = F.leaky_relu(out)
        out = F.relu(out)
        logits = self.classifier(out)
        return logits

class kmeans_classifier(nn.Module):
    def __init__(self, input_dim, num_classes,  use_bn=False):
        super(kmeans_classifier, self).__init__()
        self.use_bn = use_bn
        self.classifier = nn.Linear(in_features=input_dim, out_features=num_classes)
        self.bn_final = nn.BatchNorm1d(input_dim)
    def forward(self, U_B):
        if self.use_bn:
            U_B = self.bn_final(U_B)
        out = F.leaky_relu(U_B)
        logits = self.classifier(out)
        return logits



class passive_model(nn.Module):
    def __init__(self, input_dim, intern_dim, output_dim):
        super(passive_model, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, intern_dim),
            #nn.LeakyReLU(),
            nn.ReLU(),
            nn.Linear(intern_dim, output_dim),
            #nn.LeakyReLU(),
            nn.ReLU(),
            #nn.Linear(in_features=intern_dim, out_features=output_dim)
        )

    def forward(self, input):
        out = self.mlp(input)
        return out

class AE_model(nn.Module):
    def __init__(self, input_dim, intern_dim, output_dim):
        super(AE_model, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, intern_dim),
            nn.LeakyReLU(),
            nn.Linear(in_features=intern_dim, out_features=output_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(output_dim, intern_dim),
            nn.LeakyReLU(),
            nn.Linear(in_features=intern_dim, out_features=input_dim)
        )

    def forward(self, input):
        latent = self.encoder(input)
        out = self.decoder(latent)
        return out, latent

