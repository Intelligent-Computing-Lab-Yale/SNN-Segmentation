#---------------------------------------------------
# Imports
#---------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

cfg_features = {
    'dl-vgg9':  [64, 64, 'avg2', 128, 128, 'avg2', 256, 'atrous256', 'atrous256'],
    'dl-vgg11': [64, 'avg2', 128, 'avg2', 256, 256, 'avg2', 512, 512, 'avg1', 'atrous512', 'atrous512'],
    'dl-vgg16': [64, 64, 'avg2', 128, 128, 'avg2', 256, 256, 256, 'avg2', 512, 512, 512, 'avg1', 'atrous512', 'atrous512', 'atrous512'],
    'fcn-vgg9':  [64, 64, 'score', 'avg2', 128, 128, 'score', 'avg2', 256, 256, 256, 'score', 'avg2'],
}

cfg_classifier = {
    'dl-vgg9':  ['atrous1024', 'output'], #! don't need avg pooling?
    'dl-vgg11':  ['atrous1024', '1024-1-0', 'output'],
    'dl-vgg16':  ['atrous1024', '1024-1-0', 'output'],
    'fcn-vgg9':  ['1024-3-1', '1024-3-1', 'score'],
}

class ANN_VGG(nn.Module):
    def __init__(self, config, init='xavier'):
        super().__init__()
        
        # Architecture parameters
        self.architecture = config.architecture
        self.dataset = config.dataset
        self.img_size = config.img_size
        self.kernel_size = config.kernel_size
        self.bn = config.bn
        self.leaky = config.leaky
        self.alpha = config.alpha
        self.init = init
        
        self._make_layers()
        self._init_layers()

        if config.pretrained:
            if 'vgg11' in self.architecture:
                vgg = torchvision.models.vgg11(pretrained=True)
                state_vgg = vgg.features.state_dict()
                self.features.load_state_dict(state_vgg, strict=False)
            elif 'vgg16' in self.architecture:
                vgg = torchvision.models.vgg16(pretrained=True)
                state_vgg = vgg.features.state_dict()
                model_dict = self.features.state_dict()
                state_vgg['0.weight'] = model_dict['0.weight'] # 1. change first weight
                self.features.load_state_dict(state_vgg, strict=False) # 2. load the new state dict


    def _make_layers(self):
        bias_flag = False
        affine_flag = False
        stride = 1
        padding = (self.kernel_size-1)//2
        dilation = 2

        scores_counter = -1

        in_channels = self.dataset['input_dim']
        layer = 0
        divisor = 1
        layers, bn_layers, relu_layers = [], [], []
        self.pool_features, self.scores_features = {}, {}

        for x in (cfg_features[self.architecture]):
            if isinstance(x, str) and x[:3] == 'avg':
                self.pool_features[str(layer)] = nn.AvgPool2d(kernel_size=self.kernel_size, stride=int(x[3:]), padding=padding)
                divisor *= 2
                continue
            elif isinstance(x, str) and x[:3] == 'max':
                self.pool_features[str(layer)] = nn.MaxPool2d(kernel_size=self.kernel_size, stride=int(x[3:]), padding=padding)
                continue
            elif isinstance(x, str) and x == 'score':
                self.scores_features[str(layer)] = nn.Conv2d(in_channels, self.dataset['num_cls'], kernel_size=1)
                scores_counter += 1
                continue
            elif isinstance(x, str) and x[:6] == 'atrous':
                layers += [nn.Conv2d(in_channels, int(x[6:]), kernel_size=self.kernel_size, padding=(self.kernel_size - 1), stride=stride, dilation=dilation, bias=bias_flag)]
                in_channels = int(x[6:])
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=self.kernel_size, padding=padding, stride=stride, bias=bias_flag)]
                in_channels = x
            if self.bn:
                bn_layers += [nn.BatchNorm2d(in_channels, eps=1e-4, momentum=0.1, affine=affine_flag)]
            relu_layers += [nn.LeakyReLU(negative_slope=self.alpha, inplace=True)] if self.leaky else [nn.ReLU(inplace=True)]
            layer += 1
        
        self.features = nn.ModuleList(layers)
        self.pool_features = nn.ModuleDict(self.pool_features)
        self.scores_features = nn.ModuleDict(self.scores_features)
        if self.bn:
            self.bn_features = nn.ModuleList(bn_layers)
        self.relu_features = nn.ModuleList(relu_layers)
        
        layer = 0
        layers, bn_layers, relu_layers = [], [], []
        self.pool_classifier, self.scores_classifier = {}, {}

        for x in (cfg_classifier[self.architecture]):
            if isinstance(x, str) and x[:3] == 'avg':
                self.pool_classifier[str(layer)] = nn.AvgPool2d(kernel_size=self.kernel_size, stride=int(x[3:]), padding=padding)
                divisor *= 2
                continue
            elif isinstance(x, str) and x[:3] == 'max':
                self.pool_classifier[str(layer)] = nn.MaxPool2d(kernel_size=self.kernel_size, stride=int(x[3:]), padding=padding)
                continue
            elif isinstance(x, str) and x == 'score':
                self.scores_classifier[str(layer)] = nn.Conv2d(in_channels, self.dataset['num_cls'], kernel_size=1)
                scores_counter += 1
                continue
            elif isinstance(x, str) and x == 'output':
                layers += [nn.Conv2d(in_channels, self.dataset['num_cls'], kernel_size=1, padding=0, stride=stride, bias=bias_flag)]
                continue
            elif isinstance(x, str) and x[:6] == 'atrous':
                layers += [nn.Conv2d(in_channels, int(x[6:]), kernel_size=self.kernel_size, padding=12, stride=stride, dilation=12, bias=bias_flag)]
                in_channels = int(x[6:])
            elif isinstance(x, str):
                x = [int(val) for val in x.split('-')]
                layers += [nn.Conv2d(in_channels, x[0], kernel_size=x[1], padding=x[2], stride=stride, bias=bias_flag)]
                in_channels = x[0]
            if self.bn:
                bn_layers += [nn.BatchNorm2d(in_channels, eps=1e-4, momentum=0.1, affine=affine_flag)]
            relu_layers += [nn.LeakyReLU(negative_slope=self.alpha, inplace=True)] if self.leaky else [nn.ReLU(inplace=True)]
            layer += 1

        self.classifier = nn.ModuleList(layers)
        self.pool_classifier = nn.ModuleDict(self.pool_classifier)
        self.scores_classifier = nn.ModuleDict(self.scores_classifier)
        if self.bn:
            self.bn_classifier = nn.ModuleList(bn_layers)
        self.relu_classifier = nn.ModuleList(relu_layers)

        if 'fcn' in self.architecture:
            layers = []
            height = self.img_size[0]
            width = self.img_size[1]
            for _ in range(scores_counter):
                divisor = divisor//2
                layers += [nn.Sequential(
                    nn.ConvTranspose2d(self.dataset['num_cls'], self.dataset['num_cls'], kernel_size=self.kernel_size, stride=2, bias=False),
                    nn.UpsamplingBilinear2d((height//divisor, width//divisor))
                )]
            self.upsample = nn.ModuleList(layers)


    def _init_layers(self):
        # Initialize the weights of all the layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d)  or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                if self.init == 'xavier':
                    torch.nn.init.xavier_uniform_(m.weight, gain=2)
                elif self.init == 'kaiming':
                    if self.leaky:
                        nn.init.kaiming_normal_(m.weight, a=self.alpha, mode='fan_in', nonlinearity='leaky_relu')
                    else:
                        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

                if m.bias is not None:
                    m.bias.data.zero_()


    def forward(self, x, f=None):
        N, C, H, W = x.size()
        out = x
        spikes = []

        for k in range(len(self.features)):
            if not self.bn:
                out = self.features[k](out)
            else:
                out = self.bn_features[k](self.features[k](out))
            out = self.relu_features[k](out)

            if ('fcn' in self.architecture) and (str(k+1) in self.scores_features.keys()):
                spikes.insert(0, self.scores_features[str(k+1)](out))

            if str(k+1) in self.pool_features.keys():
                out = self.pool_features[str(k+1)](out)

        for k in range(len(self.classifier) if 'fcn' in self.architecture else (len(self.classifier) - 1)):
            if not self.bn:
                out = self.classifier[k](out)
            else:
                out = self.bn_classifier[k](self.classifier[k](out))
            out = self.relu_classifier[k](out)

            if ('fcn' in self.architecture) and (str(k+1) in self.scores_classifier.keys()):
                spikes.insert(0, self.scores_classifier[str(k+1)](out))

            if str(k+1) in self.pool_classifier.keys():
                out = self.pool_classifier[str(k+1)](out)

        if 'fcn' in self.architecture:
            out = spikes[0]

            for k in range(len(self.upsample)):
                out = self.upsample[k](out) + spikes[k+1]
        else:
            out = self.classifier[k+1](out)
            out = F.interpolate(out, (H, W), mode='bilinear', align_corners=True)

        return out