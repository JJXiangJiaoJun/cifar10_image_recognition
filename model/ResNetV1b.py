import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn

"""
ResNet model
"""


class Bottleneck(gluon.HybridBlock):
    """
        Bottlenet use in resnet,this is the basic block in resnet
    
    @:parameter
    planes : int 
        output channels of this Bottleneck (planes * expansion)
    strides : int 
        the strides of conv3x3 (default 1)
        
    down_sample : mxnet.nn.Block
        down_sample layer (default None)
        
    bn : mxnet.nn.Block
        batchnorm layer use in Bottleneck (default nn.BacthNorm)
        
    bn_kwargs : dict
        batchnorm layer argument
        
    last_gamma : boolean ,default false
        Whether to initialize the gamma of the last BatchNorm in each Bottleneck to zero
        
    """
    expansion = 4

    def __init__(self, planes, strides=1, down_sample=None,
                 norm_layer=None, norm_kwargs={}, last_gamma=False, **kwargs):
        super(Bottleneck, self).__init__(**kwargs)
        self.bn1 = norm_layer(**norm_kwargs)
        self.relu1 = nn.Activation('relu')
        self.conv1 = nn.Conv2D(channels=planes, kernel_size=1, strides=1, use_bias=False)

        self.bn2 = norm_layer(**norm_kwargs)
        self.relu2 = nn.Activation('relu')
        self.conv2 = nn.Conv2D(channels=planes, kernel_size=3, strides=strides, padding=1, use_bias=False)

        if last_gamma:
            self.bn3 = norm_layer(gamma_initializer='zeros', **norm_kwargs)
        else:
            self.bn3 = norm_layer(**norm_kwargs)

        self.relu3 = nn.Activation('relu')
        self.conv3 = nn.Conv2D(channels=planes * 4, kernel_size=1, strides=1, use_bias=False)

        self.down_sample = down_sample
        self.strides = strides

    def hybrid_forward(self, F, x, *args, **kwargs):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.down_sample is not None:
            residual = self.down_sample(x)

        out = out + residual
        out = self.relu3(out)
        return out


class ResNetV1b(gluon.HybridBlock):
    """
    @:parameter
    block : gluon.HybridBlock
        Mini-net used in ResNetV1
        
    layers : list of int
        Number blocks of each layer
    
    nn : BatchNorm
    
    bn_kwargs : dict
    
    use_global_stats : boolean
        Whether to use global stats
    
    final_drop : float
        If final_drop > 0.0 will use dropout layer
    
    last_gamma : boolean ,default false
        Whether to initialize the gamma of the last BatchNorm in each Bottleneck to zero
    
    """

    def __init__(self, block, layers, classes=10,
                 norm_layer=nn.BatchNorm, norm_kwargs={}, use_global_stats=False,
                 final_drop=0.0, last_gamma=False, name_prefix='', **kwargs):
        super(ResNetV1b, self).__init__(prefix=name_prefix, **kwargs)

        self.inplanes = 64
        self.norm_kwargs = norm_kwargs
        if use_global_stats:
            self.norm_kwargs['use_global_stats'] = True
        self._classes = classes
        with self.name_scope():
            self.bn1 = norm_layer(**norm_kwargs)
            self.relu1 = nn.Activation('relu')
            self.conv1 = nn.Conv2D(channels=64, kernel_size=5, strides=2, padding=2, use_bias=False)

            self.maxpool = nn.MaxPool2D(pool_size=3, strides=2, padding=1)

            self.layer1 = self._make_layer(1, block, layers[0], 64,
                                           strides=1, norm_layer=norm_layer, last_gamma=last_gamma)
            self.layer2 = self._make_layer(2, block, layers[1], 128,
                                           strides=2, norm_layer=norm_layer, last_gamma=last_gamma)
            self.layer3 = self._make_layer(3, block, layers[2], 256,
                                           strides=2, norm_layer=norm_layer, last_gamma=last_gamma)
            self.layer4 = self._make_layer(4, block, layers[3], 512,
                                           strides=2, norm_layer=norm_layer, last_gamma=last_gamma)

            self.avg_pool = nn.GlobalAvgPool2D()
            self.flat = nn.Flatten()
            self.drop = None
            if final_drop > 0.0:
                self.drop = nn.Dropout(final_drop)
            self.fc = nn.Dense(in_units=512 * block.expansion, units=classes)

    def _make_layer(self, stage_index, block, layers, planes, strides=1, norm_layer=None, last_gamma=False):

        down_sample = None
        if strides != 1 or self.inplanes != planes * block.expansion:
            # down_sample
            down_sample = nn.HybridSequential(prefix='down{}_'.format(stage_index))
            down_sample.add(nn.AvgPool2D(pool_size=3, strides=strides, padding=1),  # down sample
                            nn.Conv2D(planes * block.expansion, kernel_size=1, strides=1, use_bias=False))
            down_sample.add(norm_layer(**self.norm_kwargs))

        layer = nn.HybridSequential(prefix='layer{}_'.format(stage_index))

        with layer.name_scope():
            layer.add(block(planes=planes, strides=strides, down_sample=down_sample,
                            norm_layer=norm_layer, norm_kwargs=self.norm_kwargs,
                            last_gamma=last_gamma))

            for _ in range(0, layers - 1):
                layer.add(block(planes=planes, norm_layer=norm_layer,
                                norm_kwargs=self.norm_kwargs, last_gamma=last_gamma))

        return layer

    def hybrid_forward(self, F, x, *args, **kwargs):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avg_pool(x)
        x = self.flat(x)

        if self.drop is not None:
            x = self.drop(x)
        x = self.fc(x)

        return x


resnet50_layers = [3, 4, 6, 3]


def resnet50_cifar10(block=Bottleneck, layers=resnet50_layers, classes=10, name_prefix='resnet50v1', **kwargs):
    net = ResNetV1b(block=block, layers=layers, classes=classes,
                    name_prefix=name_prefix, **kwargs)

    return net
