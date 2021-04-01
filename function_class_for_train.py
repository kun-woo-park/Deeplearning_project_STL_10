import time
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset
from typing import Type, Any, Callable, Union, List, Optional
from torch import Tensor


def train_data_load(train_path_path, val_path_path, batch_size, resolution, normalize, num_of_data_augmentation = 100000):
    config_transform = transforms.Compose([
            transforms.RandomRotation(degrees=45),

            transforms.RandomResizedCrop(resolution),
            transforms.ColorJitter(.3, .3, .3, .3),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    augmentation_loop = int(num_of_data_augmentation / 5000) - 1

    # data augmentation
    train_set = datasets.ImageFolder(
        train_path_path, config_transform
        )

    val_set = datasets.ImageFolder(
        val_path_path,
        config_transform
    )
    for _ in range(augmentation_loop):
        train_aug_data = datasets.ImageFolder(
            train_path_path,
            config_transform
        )

        val_aug_data = datasets.ImageFolder(
            val_path_path,
            config_transform
        )
        train_set = torch.utils.data.ConcatDataset([train_set, train_aug_data])
        val_set = torch.utils.data.ConcatDataset([val_set, val_aug_data])

    # generate data loader
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=8, pin_memory=True, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=True,
        num_workers=8, pin_memory=True, drop_last=True)

    return train_loader, val_loader


def train_model(model, total_epoch, val_acc_list, batch_size, lr, lr_gamma, patience, start_early_stop_check,
                saving_start_epoch, model_char, saving_path, T_0, T_mul, train_loader, val_loader):

    # set loss, optimizer and lr scheduler
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_0, eta_min=0, last_epoch=-1)

    sch_step = 0
    lr_list = []
    trn_loss_list = []
    val_loss_list = []
    model_name = ""

    for epoch in range(total_epoch):
        trn_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()
            # grad init
            optimizer.zero_grad()
            # forward propagation
            output = model(inputs)
            # calculate loss
            loss = criterion(output, labels)
            # back propagation
            loss.backward()
            # weight update
            optimizer.step()

            # trn_loss summary
            trn_loss += loss.item()
        # validation
        with torch.no_grad():
            val_loss = 0.0
            cor_match = 0
            for j, val in enumerate(val_loader):
                val_x, val_label = val
                if torch.cuda.is_available():
                    val_x = val_x.cuda()
                    val_label = val_label.cuda()
                val_output = model(val_x)
                v_loss = criterion(val_output, val_label)
                val_loss += v_loss
                _, predicted = torch.max(val_output, 1)
                cor_match += np.count_nonzero(predicted.cpu().detach()
                                              == val_label.cpu().detach())
        # step customized lr scheduler
        if sch_step == T_0:
            sch_step = 0
            T_0 *= T_mul
            optimizer.param_groups[0]['initial_lr'] = lr*lr_gamma
            lr = lr*lr_gamma
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_0, eta_min=0, last_epoch=-1)
        else:
            scheduler.step()
            lr_list.append(optimizer.param_groups[0]['lr'])
            sch_step += 1

        trn_loss_list.append(trn_loss/len(train_loader))
        val_loss_list.append(val_loss/len(val_loader))
        val_acc = cor_match/(len(val_loader)*batch_size)
        val_acc_list.append(val_acc)
        now = time.localtime()
        print("%04d/%02d/%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon,
              now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec))

        print("epoch: {}/{} | trn loss: {:.4f} | val loss: {:.4f} | val accuracy: {:.4f}% \n".format(
            epoch+1, total_epoch, trn_loss /
            len(train_loader), val_loss / len(val_loader), val_acc*100
        ))

        # early stop
        if epoch+1 > 2:
            if val_loss_list[-1] > val_loss_list[-2]:
                start_early_stop_check = 1
        else:
            val_loss_min = val_loss_list[-1]

        if start_early_stop_check:
            early_stop_temp = val_loss_list[-patience:]
            if all(early_stop_temp[i] < early_stop_temp[i+1] for i in range(len(early_stop_temp)-1)):
                print("Early stop!")
                break

        # save the minimum loss model
        if epoch+1 > saving_start_epoch:
            if val_loss_list[-1] < val_loss_min:
                if os.path.isfile(model_name):
                    os.remove(model_name)
                val_loss_min = val_loss_list[-1]
                model_name = saving_path+"Custom_model_" + \
                    model_char+"_{:.3f}".format(val_loss_min)
                torch.save(model, model_name)
                print("Model replaced and saved as ", model_name)


########## Using ResNet Open Source (change some parameters manually) ###########
#################################################################################


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.LeakyReLU(0.1)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(
        self,
        block: Type[Union[Bottleneck]],
        layers: List[int],
        num_classes: int = 10,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=4, stride=1,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.LeakyReLU(0.1)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer1 = self._make_layer(block, 30, layers[0])
        self.layer2 = self._make_layer(block, 60, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 96, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(96 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    # type: ignore[arg-type]
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block: Type[Union[Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
    arch: str,
    block: Type[Union[Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def Model(pretrained: bool = False, progress: bool = True, **kwargs):

    kwargs['groups'] = 1
    kwargs['width_per_group'] = 64
    return _resnet('resnet', Bottleneck, [4, 9, 8], pretrained, progress, **kwargs)

#################################################################################
#################################################################################
