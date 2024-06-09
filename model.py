import torch
import torch.nn. functional as F
import torch.nn as nn
from torchvision.models import resnet34


class PreFixResnet(nn.Module):

    def __init__(self, use_channel, train):

        super().__init__()

        in_channel = len(use_channel)
        self.prefix = Dynamic_conv2d(in_channel, out_channel=3, kernel_size=1,num_kernel=8)
        
        # resnet34 is backbone, so i always frozen resnet34 parameter
        self.resnet = resnet34()
        self.resnet.fc = nn.Linear(512, 50)
        if train:
            self.resnet.load_state_dict(torch.load('./ckpt/06_08_16_naive_RGB.pt'))
        # frozen resnet34 parameter
        for weight in self.resnet.parameters():
            weight.requires_grad = False
        
    def forward(self, img):

        x = self.prefix(img) # (bs, in_c, H, W) -> (bs, 3, H, W)
        out = self.resnet(x)

        return out
    


# https://github.com/kaijieshi7/Dynamic-convolution-Pytorch/blob/master/dynamic_conv.py

class attention2d(nn.Module):

    def __init__(self, in_planes, ratios, num_kernel):

        super(attention2d, self).__init__()

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        if in_planes!=3:
            hidden_planes = int(in_planes*ratios)+1
        else:
            hidden_planes = num_kernel

        self.fc1 = nn.Conv2d(in_planes, hidden_planes, 1, bias=False)
        # self.bn = nn.BatchNorm2d(hidden_planes)
        self.fc2 = nn.Conv2d(hidden_planes, num_kernel, 1, bias=True)


        self._initialize_weights()


    def _initialize_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m ,nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, temperature):

        x = self.avgpool(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x).view(x.size(0), -1)

        return F.softmax(x/temperature, 1)


class Dynamic_conv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, ratio=0.25, stride=1, padding=0, dilation=1, groups=1, bias=True, num_kernel=4):
        super(Dynamic_conv2d, self).__init__()

        assert in_channel%groups==0

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = num_kernel
        self.attention = attention2d(in_channel, ratio, num_kernel)

        self.weight = nn.Parameter(torch.randn(num_kernel, out_channel, in_channel//groups, kernel_size, kernel_size), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.zeros(num_kernel, out_channel))
        else:
            self.bias = None

        self._initialize_weights()

    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])


    def forward(self, x, temperature = 1):
        
        batch_size, in_channel, height, width = x.size()

        
        softmax_attention = self.attention(x, temperature)
        
        x = x.view(1, -1, height, width)
        weight = self.weight.view(self.K, -1) # (K, out_c*in_c*ks*ks)

        # bs*out_c X in_c X Kernel_size X Kernel_size
        aggregate_weight = torch.mm(softmax_attention, weight).view(batch_size*self.out_channel, self.in_channel//self.groups, self.kernel_size, self.kernel_size)
        
        if self.bias is not None:
            aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups*batch_size)
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)

        output = output.view(batch_size, self.out_channel, output.size(-2), output.size(-1))

        return output