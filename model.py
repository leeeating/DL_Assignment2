import torch
import torch.nn. functional as F
import torch.nn as nn
from torchvision.models import resnet34


class PreFixResnet(nn.Module):

    def __init__(self, use_channel, train):

        super().__init__()

        in_channel = len(use_channel)
        self.prefix = Dynamic_conv2d(in_channel, out_channel=3, kernel_size=1, num_kernel=8)
        
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
    


class CustomModel(nn.Module):

    def __init__(self, in_c=3, out_class=50):

        super(CustomModel,self).__init__()

        #(bs,3,224,224)
        self.dycnn1 = Dynamic_conv2d(in_channel=in_c, out_channel=64, kernel_size=7, stride=2, padding=3, num_kernel=8, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        #(bs,64,112,112)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        #(bs,64,56,56)
        self.dycnn2 = Dynamic_conv2d(in_channel=64, out_channel=128, kernel_size=3,stride=2, padding=1, num_kernel=8, bias=False)
        self.bn2 = nn.BatchNorm2d(128)

        #(bs,128,28,28)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        #(bs,128,14,14)
        self.resblock = self._make_layer(ResidualBlock, 256, blocks=1, first_stride=2)

        #(bs,256,7,7)
        self.atten = SelfAttention(in_channels=256, heads=4, head_size=256)

        ##(bs,256,7,7)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        #(bs,256,1)
        self.fc = nn.Linear(256, out_class)

    def _make_layer(self, block, out_channels, blocks, first_stride):
        
        shortcut = None

        if first_stride != 1 :
            shortcut = nn.Sequential(
                nn.Conv2d(128, out_channels, kernel_size=1, stride=first_stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = []
        layers.append(block(128, out_channels, first_stride, shortcut))

        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, imgs):

        N = imgs.shape[0]

        x = self.dycnn1(imgs)
        x = self.bn1(x)
        x = self.pool1(x)

        x = self.dycnn2(x)
        x = self.bn2(x)
        x = self.pool2(x)

        x = self.resblock(x)

        x = self.atten(x)
        x = self.global_pool(x).view(N,256)
        
        out = self.fc(x)        
        
        return out
    





class ResidualBlock(nn.Module): 
  
    def __init__(self, in_channel, out_channel, stride=1, shortcut=None):         
        super(ResidualBlock, self).__init__() 
        self.left = nn.Sequential(   
        nn.Conv2d(in_channel, out_channel, 3, stride, 1, bias=False), # bias=False是因為bias再BN中已經有了，如果stride=2則shape會變成一半 
        nn.BatchNorm2d(out_channel), nn.ReLU(), 
        nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False), # shape前後仍然相同 
        nn.BatchNorm2d(out_channel), 
        ) 
        self.right = shortcut # 根據情況是否做出增維或是縮小shape 

    def forward(self, x): 

        out = self.left(x) 
        residual = x if self.right is None else self.right(x) 
        out = out + residual 
        out = F.relu(out) 
        return out


class SelfAttention(nn.Module):
    def __init__(self, in_channels, heads, head_size):

        super(SelfAttention, self).__init__()
        
        self.heads = heads
        self.head_dim = head_size
        self.embed_size = head_size*heads

        self.values = nn.Conv2d(in_channels, self.embed_size, kernel_size=1, bias=False)
        self.keys = nn.Conv2d(in_channels, self.embed_size, kernel_size=1, bias=False)
        self.queries = nn.Conv2d(in_channels, self.embed_size, kernel_size=1, bias=False)
        self.fc_out = nn.Conv2d(self.embed_size, in_channels, kernel_size=1)

    def forward(self, imgs):

        batch_size,_ ,H, W = imgs.shape

        values = self.values(imgs).view(batch_size, self.heads, -1, self.head_dim)
        keys = self.keys(imgs).view(batch_size, self.heads, -1, self.head_dim)
        queries = self.queries(imgs).view(batch_size, self.heads, -1, self.head_dim)

        # 使用 einsum 計算點積注意力
        energy = torch.einsum("nhqd,nhkd->nhqk", [queries, keys])
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhqk,nhqd->nhqd", [attention, values])
        out = out.view(batch_size, self.embed_size, H, W)
        out = self.fc_out(out)

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
        
        batch_size, _, height, width = x.size()

        
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