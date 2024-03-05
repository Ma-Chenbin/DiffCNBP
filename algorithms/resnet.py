import torch.nn as nn
import torch
from functools import partial
import torchvision.models as models
from torchsummary import summary

class Conv1dAuto(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding = (self.kernel_size[0]//2,)
        self.bias = None

class dpConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, bias, kernel_size=5, stride=1):
        super(dpConv1d, self).__init__()
        self.dw_conv = nn.Conv1d(in_channels=in_ch,
                                 out_channels=in_ch,
                                 kernel_size=kernel_size,
                                 stride=1,
                                 padding=int(kernel_size/2),
                                 groups=in_ch,
                                 bias=False)
        self.pw_conv = nn.Conv1d(in_channels=in_ch,
                                 out_channels=out_ch,
                                 kernel_size=1,
                                 stride=stride,
                                 padding=0,
                                 bias=False)
    def forward(self, input):
        output = self.dw_conv(input)
        output = self.pw_conv(output)
        # print(output.shape)
        return output

Conv1d = partial(Conv1dAuto, kernel_size=3)

# Conv1d = partial(dpConv1d, kernel_size=3)

def activation_func(activation):
    return nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['none', nn.Identity()]
    ])[activation]

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu', *args, **kwargs):
        super(ResidualBlock, self).__init__()
        self.in_channels, self.out_channels, self.activation = in_channels, out_channels, activation
        self.blocks = nn.Identity()
        self.activate = activation_func(activation)
        self.shortcut = nn.Identity()

    def forward(self, x):
        residual = x
        if self.should_apply_shortcut: residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        x = self.activate(x)
        return x

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels

class ResNetResidualBlock(ResidualBlock):
    def __init__(self, in_channels, out_channels, expansion=1, downsampling=1, conv=Conv1d, *args, **kwargs):
        super(ResNetResidualBlock, self).__init__(in_channels, out_channels, *args, **kwargs)
        self.expansion, self.downsampling, self.conv = expansion, downsampling, conv
        self.shortcut = nn.Sequential(
            nn.Conv1d(self.in_channels, self.expanded_channels, kernel_size=1, stride=self.downsampling, bias=False),
            nn.BatchNorm1d(self.expanded_channels) if 'bn' in kwargs.keys() and kwargs['bn'] == True else nn.Identity()
        ) if self.should_apply_shortcut else None

    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.expanded_channels

def conv_bn(in_channels, out_channels, conv, *args, **kwargs):
    return nn.Sequential(conv(in_channels, out_channels, *args, **kwargs),
                         nn.BatchNorm1d(out_channels) if 'bn' in kwargs.keys() and kwargs['bn'] == True else nn.Identity())

class ResNetBasicBlock(ResNetResidualBlock):
    """
    Basic ResNet block composed by two layers conv1d-->bn-->activation
    """
    expansion = 1
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels, conv=self.conv, bias=False, stride=self.downsampling),
            activation_func(self.activation),
            conv_bn(self.out_channels, self.out_channels, conv=self.conv, bias=False)
        )

class ResNetBottleNeckBlock(ResNetResidualBlock):
    expansion = 4
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(in_channels, out_channels, expansion=4, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels, self.conv, kernel_size=1),
            activation_func(self.activation),
            conv_bn(self.out_channels, self.out_channels, self.conv, kernel_size=3, stride=self.downsampling),
            activation_func(self.activation),
            conv_bn(self.out_channels, self.expanded_channels, self.conv, kernel_size=1)
        )

class ResNetLayer(nn.Module):
    """
    A ResNet Layer composed by n blocks stacked one after the other
    """
    def __init__(self, in_channels, out_channels, block=ResNetBasicBlock, n=1, *args, **kwargs):
        super().__init__()
        if in_channels == out_channels:
            downsampling = 1
        else:
            downsampling = 2 if not 'downsampling' in kwargs.keys() else kwargs['downsampling']
        kwargs.__delitem__('downsampling')
        self.blocks = nn.Sequential(
            block(in_channels, out_channels, *args, **kwargs, downsampling=downsampling),
            *[block(out_channels*block.expansion,
                    out_channels,
                    downsampling=1, *args, **kwargs) for _ in range(n-1)]
        )

    def forward(self, x):
        x = self.blocks(x)
        return x

class myResNet_K(nn.Module):
    def __init__(self, in_channels, block1_size=64, deepths=[2,2,2,2],
                 activation='relu', block=ResNetBasicBlock, dropout=0.5, *args, **kwargs):
        super(myResNet_K, self).__init__()
        # print(kwargs)
        self.block1_size = block1_size
        self.gate = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=self.block1_size,
                      kernel_size=5, stride=2, padding=2, bias=False),
            # nn.BatchNorm1d(self.block1_size) if 'bn' in kwargs.keys() and kwargs['bn'] == True else nn.Identity(),
            activation_func(activation),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        self.in_out_blocks_size = list((self.block1_size*pow(2, i), self.block1_size*pow(2, i+1)) for i in range(0, 3))
        self.blocks = nn.Sequential(
            ResNetLayer(self.block1_size, self.block1_size, n=deepths[0], activation=activation,
                        block=block, *args, **kwargs),
            *[ResNetLayer(in_channels*block.expansion, out_channels, n=n, activation=activation, block=block, *args, **kwargs)
              for (in_channels, out_channels), n in zip(self.in_out_blocks_size, deepths[1:-1])],
            ResNetLayer(self.in_out_blocks_size[-1][0]*block.expansion, self.in_out_blocks_size[-1][1], n=deepths[-1],
                        activation='none', block=block, *args, **kwargs)
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.linear1 = nn.Linear(block1_size*8+10, block1_size*8+10, bias=False)
        self.linear2 = nn.Linear(block1_size*8+10, 1, bias=False)
        # self.bn = nn.BatchNorm1d(block1_size*8+10)

    def forward(self, x, info=None):
        if isinstance(info, type(None)):
            info = torch.randn(len(x), 10).cuda()
        x = self.gate(x)
        x = self.blocks(x)
        x = self.gap(x).squeeze(dim=-1)
        x = torch.cat((x, info), dim=1)
        x = self.linear1(x)
        x = self.linear2(x)
        return x

class myResNet(nn.Module):
    def __init__(self, in_channels, block1_size=64, deepths=[2,2,2,2],
                 activation='relu', block=ResNetBasicBlock, dropout=0.5, *args, **kwargs):
        super(myResNet, self).__init__()
        # print(kwargs)
        self.block1_size = block1_size
        self.gate = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=self.block1_size,
                      kernel_size=5, stride=2, padding=2, bias=False),
            # nn.BatchNorm1d(self.block1_size) if 'bn' in kwargs.keys() and kwargs['bn'] == True else nn.Identity(),
            activation_func(activation),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        self.in_out_blocks_size = list((self.block1_size*pow(2, i), self.block1_size*pow(2, i+1)) for i in range(0, 3))
        self.blocks = nn.Sequential(
            ResNetLayer(self.block1_size, self.block1_size, n=deepths[0], activation=activation,
                        block=block, *args, **kwargs),
            *[ResNetLayer(in_channels*block.expansion, out_channels, n=n, activation=activation, block=block, *args, **kwargs)
              for (in_channels, out_channels), n in zip(self.in_out_blocks_size, deepths[1:-1])],
            ResNetLayer(self.in_out_blocks_size[-1][0]*block.expansion, self.in_out_blocks_size[-1][1], n=deepths[-1],
                        activation='none', block=block, *args, **kwargs)
        )
        self.gap = nn.AdaptiveAvgPool1d(1)

    def forward(self, x, info=None):
        x = self.gate(x)
        x = self.blocks(x)
        x = self.gap(x).squeeze(dim=-1)
        x = torch.cat((x, info), dim=1)
        return x

class myResNet_info(nn.Module):
    def __init__(self, in_channels, block1_size=64, deepths=[2,2,2,2],
                 activation='relu', block=ResNetBasicBlock, dropout=0.5, *args, **kwargs):
        super(myResNet_info, self).__init__()
        # print(kwargs)
        self.block1_size = block1_size
        self.gate = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=self.block1_size,
                      kernel_size=5, stride=2, padding=2, bias=False),
            # nn.BatchNorm1d(self.block1_size) if 'bn' in kwargs.keys() and kwargs['bn'] == True else nn.Identity(),
            activation_func(activation),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        self.in_out_blocks_size = list((self.block1_size*pow(2, i), self.block1_size*pow(2, i+1)) for i in range(0, 3))
        self.blocks = nn.Sequential(
            ResNetLayer(self.block1_size, self.block1_size, n=deepths[0], activation=activation,
                        block=block, *args, **kwargs),
            *[ResNetLayer(in_channels*block.expansion, out_channels, n=n, activation=activation, block=block, *args, **kwargs)
              for (in_channels, out_channels), n in zip(self.in_out_blocks_size, deepths[1:-1])],
            ResNetLayer(self.in_out_blocks_size[-1][0]*block.expansion, self.in_out_blocks_size[-1][1], n=deepths[-1],
                        activation='none', block=block, *args, **kwargs)
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(block1_size*8+10, block1_size*8+10, bias=False)

    def forward(self, x, info=None):
        if isinstance(info, type(None)):
            info = torch.randn(len(x), 10).cuda()
        x = self.gate(x)
        x = self.blocks(x)
        x = self.gap(x).squeeze(dim=-1)
        x = torch.cat((x, info), dim=1)
        x = self.linear(x)
        return x

class myResNet_hr(nn.Module):
    def __init__(self, in_channels, block1_size=64, deepths=[2,2,2,2],
                 activation='relu', block=ResNetBasicBlock, dropout=0.5, *args, **kwargs):
        super(myResNet_hr, self).__init__()
        # print(kwargs)
        self.block1_size = block1_size
        self.gate = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=self.block1_size,
                      kernel_size=5, stride=2, padding=2, bias=False),
            # nn.BatchNorm1d(self.block1_size) if 'bn' in kwargs.keys() and kwargs['bn'] == True else nn.Identity(),
            activation_func(activation),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        self.in_out_blocks_size = list((self.block1_size*pow(2, i), self.block1_size*pow(2, i+1)) for i in range(0, 3))
        self.blocks = nn.Sequential(
            ResNetLayer(self.block1_size, self.block1_size, n=deepths[0], activation=activation,
                        block=block, *args, **kwargs),
            *[ResNetLayer(in_channels*block.expansion, out_channels, n=n, activation=activation, block=block, *args, **kwargs)
              for (in_channels, out_channels), n in zip(self.in_out_blocks_size, deepths[1:-1])],
            ResNetLayer(self.in_out_blocks_size[-1][0]*block.expansion, self.in_out_blocks_size[-1][1], n=deepths[-1],
                        activation='none', block=block, *args, **kwargs)
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        # self.linear = nn.Linear(block1_size*8+11, block1_size*8+11, bias=False)
        # self.bn_features = nn.BatchNorm1d(block1_size*8+13)

    def forward(self, x, info=None):
        if isinstance(info, type(None)):
            info = torch.randn(len(x), 11).cuda()
        x = self.gate(x)
        x = self.blocks(x)
        x = self.gap(x).squeeze(dim=-1)
        x = torch.cat((x, info), dim=1)
        # x = self.linear(x)
        # x = self.bn_features(x)
        return x

class myResNet_hr_xgb(nn.Module):
    def __init__(self, in_channels, block1_size=64, deepths=[2,2,2,2], xgb_dims=10,
                 activation='relu', block=ResNetBasicBlock, dropout=0.5, *args, **kwargs):
        super(myResNet_hr_xgb, self).__init__()
        # print(kwargs)
        self.block1_size = block1_size
        self.gate = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=self.block1_size,
                      kernel_size=5, stride=2, padding=2, bias=False),
            # nn.BatchNorm1d(self.block1_size) if 'bn' in kwargs.keys() and kwargs['bn'] == True else nn.Identity(),
            activation_func(activation),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        self.in_out_blocks_size = list((self.block1_size*pow(2, i), self.block1_size*pow(2, i+1)) for i in range(0, 3))
        self.blocks = nn.Sequential(
            ResNetLayer(self.block1_size, self.block1_size, n=deepths[0], activation=activation,
                        block=block, *args, **kwargs),
            *[ResNetLayer(in_channels*block.expansion, out_channels, n=n, activation=activation, block=block, *args, **kwargs)
              for (in_channels, out_channels), n in zip(self.in_out_blocks_size, deepths[1:-1])],
            ResNetLayer(self.in_out_blocks_size[-1][0]*block.expansion, self.in_out_blocks_size[-1][1], n=deepths[-1],
                        activation='none', block=block, *args, **kwargs)
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(block1_size*8, xgb_dims, bias=False)
        # self.bn_features = nn.BatchNorm1d(block1_size*8+13)

    def forward(self, x, info=None):
        if isinstance(info, type(None)):
            info = torch.randn(len(x), 11).cuda()
        x = self.gate(x)
        x = self.blocks(x)
        x = self.gap(x).squeeze(dim=-1)
        x = self.linear(x)
        x = torch.cat((x, info), dim=1)
        # x = self.bn_features(x)
        return x

class myResNet_info_internal(nn.Module):
    def __init__(self, in_channels, block1_size=64, deepths=[2,2,2,2],
                 activation='relu', block=ResNetBasicBlock, dropout=0.5, *args, **kwargs):
        super(myResNet_info_internal, self).__init__()
        # print(kwargs)
        self.block1_size = block1_size
        self.gate = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=self.block1_size,
                      kernel_size=5, stride=2, padding=2, bias=False),
            # nn.BatchNorm1d(self.block1_size) if 'bn' in kwargs.keys() and kwargs['bn'] == True else nn.Identity(),
            activation_func(activation),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        self.in_out_blocks_size = list((self.block1_size*pow(2, i), self.block1_size*pow(2, i+1)) for i in range(0, 3))
        self.blocks = nn.Sequential(
            ResNetLayer(self.block1_size, self.block1_size, n=deepths[0], activation=activation,
                        block=block, *args, **kwargs),
            *[ResNetLayer(in_channels*block.expansion, out_channels, n=n, activation=activation, block=block, *args, **kwargs)
              for (in_channels, out_channels), n in zip(self.in_out_blocks_size, deepths[1:-1])],
            ResNetLayer(self.in_out_blocks_size[-1][0]*block.expansion, self.in_out_blocks_size[-1][1], n=deepths[-1],
                        activation='none', block=block, *args, **kwargs)
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(block1_size*8+10, block1_size*8+10, bias=False)
        # self.bn = nn.BatchNorm1d(block1_size*8+10)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, info=None):
        if isinstance(info, type(None)):
            info = torch.randn(len(x), 10).cuda()
        x = self.gate(x)
        x = self.blocks(x)
        x = self.gap(x).squeeze(dim=-1)
        x = torch.cat((x, info), dim=1)
        intermediate = x
        x = self.linear(x)
        # x = self.bn(x)
        x = self.dropout(x)
        return x, intermediate

if __name__ == '__main__':
    pre_model = myResNet_hr(in_channels=2, block1_size=8, bn=True, deepths=[2,2,2,2], downsampling=3).cuda()
    summary(pre_model, (2,30*128))
