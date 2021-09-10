from re import X
from torch import nn
import torch
from torch.autograd.grad_mode import F
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
from torch.nn import init


from torch import autograd
import math

__all__ = ['MobileNetV2', 'mobilenet_v2']


model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class TopKBinarizer(autograd.Function):
    """
    Top-k Binarizer.
    Computes a binary mask M from a real value matrix S such that `M_{i,j} = 1` if and only if `S_{i,j}`
    is among the k% highest values of S.

    Implementation is inspired from:
        https://github.com/allenai/hidden-networks
        What's hidden in a randomly weighted neural network?
        Vivek Ramanujan*, Mitchell Wortsman*, Aniruddha Kembhavi, Ali Farhadi, Mohammad Rastegari
    """

    @staticmethod
    def forward(ctx, inputs: torch.tensor, sparsity: float, block_size : int):
        mask = inputs.clone()
        block_mask =  mask.reshape(block_size, -1)
        block_shape = block_mask.shape
        block_mask = torch.sum(block_mask, dim=0)
        _, idx = block_mask.sort(descending=True)
        j = int(sparsity * block_mask.numel())
        block_mask[idx[j:]] = 0
        block_mask[idx[:j]] = 1
        block_mask = block_mask.unsqueeze(0).expand(block_shape)
        block_mask = block_mask.reshape(inputs.shape)
        return block_mask

    @staticmethod
    def backward(ctx, gradOutput):
        return gradOutput, None, None


class SparseDepthwiseConv(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size = 1,
            stride = 1,
            padding = 0,
            dilation = 1,
            groups = 1,
            bias = True,
            padding_mode: str = 'zeros',
            sparsity = 0.5, 
            block_size = 4, 
        ):
        super(SparseDepthwiseConv, self).__init__()
        assert(kernel_size == 1)
        assert(padding == 0)
        assert(stride == 1)
        assert(dilation == 1)
        assert(groups == 1)

        self.sparsity = sparsity
        self.block_size = block_size
        self.add_bias = bias

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.zeros(self.out_channels, self.in_channels))

        if self.add_bias:
            self.bias = nn.Parameter(torch.zeros(self.out_channels))
        
        assert(self.out_channels % block_size == 0)
        self.should_prune = False
    
    def init_mask(self):
        init.kaiming_uniform_(self.mask_scores, a=math.sqrt(5))

    def block_topK_binarizer(self, inputs):
        # inputs: out_channel * in_channel
        mask = inputs.clone()
        block_mask =  mask.reshape(self.block_size, -1)
        block_shape = block_mask.shape
        block_mask = torch.sum(block_mask, dim=0)
        _, idx = block_mask.sort(descending=True)
        j = int(self.sparsity * block_mask.numel())
        block_mask[idx[j:]] = 0
        block_mask[idx[:j]] = 1
        block_mask = block_mask.unsqueeze(0).expand(block_shape)
        block_mask = block_mask.reshape(inputs.shape)
        return block_mask

    def forward(self, x):
        if self.should_prune:
            mask = self.block_topK_binarizer(torch.abs(self.weight))
            self.weight.data = self.weight.data * mask
            self.should_prune = False

        x = x.permute(0,2,3,1)
        x = torch.matmul(x, self.weight.T)
        x = x.permute(0,3,1,2)

        if self.add_bias: 
            x = x + self.bias

        return x



class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=None, sparsity = 1, block_size = 1):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        super(ConvBNReLU, self).__init__(
            # check whether to use sparse 1x1 conv
            SparseDepthwiseConv(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False, sparsity = sparsity, block_size = block_size) if kernel_size == 1 and sparsity < 1\
                else nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, norm_layer=None, sparsity = 1, block_size = 1):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer, sparsity = sparsity))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
            # pw-linear
            SparseDepthwiseConv(hidden_dim, oup, 1, 1, 0, bias=False, sparsity = sparsity, block_size = block_size) if (sparsity < 1) else \
                  nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self,
                 num_classes=1000,
                 width_mult=1.0,
                 inverted_residual_setting=None,
                 round_nearest=8,
                 block=None,
                 norm_layer=None,
                 sparsity = 1, 
                 block_size = 1):
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use

        """
        super(MobileNetV2, self).__init__()

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2, norm_layer=norm_layer)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer, sparsity = sparsity, block_size = block_size))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer, sparsity = sparsity, block_size = block_size))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x):
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1 => must use reshape with x.shape[0]
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)

def mobilenet_v2(pretrained=False, progress=True, **kwargs):
    """
    Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MobileNetV2(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

def sparse_mobilenet_v2(pretrained=False, progress=True, sparsity = 0.5, block_size = 1, **kwargs):
    model = MobileNetV2(sparsity = sparsity, block_size = block_size, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'],
                                              progress=progress)
        # model.load_state_dict(state_dict)
        for n, p in state_dict.items():
            try:
                model.state_dict()[n].data.copy_(p)
            except:
                p = p.squeeze(-1).squeeze(-1)
                model.state_dict()[n].data.copy_(p)
    return model

if __name__ == "__main__":
    sparsity = 0.1
    block_size = 1
    model = sparse_mobilenet_v2(pretrained=True, sparsity = sparsity, block_size = block_size)
    # test 
    x = torch.zeros((16, 3, 224,224))
    y = model(x)
    # save model 
    # torch.save(model.state_dict(),"sparse_mobilenet_v2_s{}_b{}".format(sparsity, block_size))
