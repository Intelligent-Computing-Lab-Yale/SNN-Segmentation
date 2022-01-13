#---------------------------------------------------
# Imports
#---------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from utils.scores import scores

from .spikes import *
from .net_utils import AverageMeterNetwork

cfg_features = {
    'dl-vgg9':  [64, 64, 'avg2', 128, 128, 'avg2', 256, 'atrous256', 'atrous256'],
    'fcn-vgg9':  [64, 64, 'score', 'avg2', 128, 128, 'score', 'avg2', 256, 256, 256, 'score', 'avg2'],
}

cfg_classifier = {
    'dl-vgg9':  ['atrous1024', 'output'], #! don't need avg pooling?
    'fcn-vgg9':  ['1024-3-1', '1024-3-1', 'score'],
}

class SNN_VGG(nn.Module):
    def __init__(self, config, grad_type='Linear', init='xavier'):
        super(SNN_VGG, self).__init__()

        self.gpu = config.gpu

        # Architecture parameters
        self.architecture = config.architecture
        self.dataset = config.dataset
        self.img_size = config.img_size
        self.kernel_size = config.kernel_size
        self.bntt = config.bn
        self.ibt = config.ibt
        self.alpha = config.alpha
        self.init = init
        self.fcn = 'fcn' in self.architecture

        # SNN simulation parameters
        self.timesteps = config.timesteps
        self.leak_mem = torch.tensor(config.leak_mem)
        self.def_threshold = config.def_threshold
        self.upsample_threshold = config.def_threshold
        self.thl = config.thl
        self.full_prop = config.full_prop
        self.count_spikes = config.count_spikes

        # Instantiate differentiable spiking nonlinearity
        self.spike_fn = init_spike_fn(grad_type)
        self.input_layer = PoissonGenerator(self.gpu)

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

        if self.count_spikes:
            # Make AverageMeterNetwork for measuring spikes
            model_length = len(self.features) + len(self.classifier) - 1
            self.activations = AverageMeterNetwork(model_length)

            if self.fcn:
                mem_features, mem_classifier, mem_upsample = self._initialize_mems(1)
            else:
                mem_features, mem_classifier = self._initialize_mems(1)
            for i in range(model_length):
                if i < len(mem_features):
                    layer_shape = mem_features[i].shape
                elif i < len(mem_features) + len(mem_classifier):
                    layer_shape = mem_classifier[i].shape
                else:
                    layer_shape = mem_upsample[i].shape
                neurons = layer_shape[1] * layer_shape[2] * layer_shape[3]
                self.activations.updateUnits(i, neurons)


    def _make_layers(self):
        affine_flag = True
        bias_flag = False
        stride = 1
        padding = (self.kernel_size-1)//2
        dilation = 2

        self.scores_counter = -1

        in_channels = self.dataset['input_dim']
        divisor = 1
        layer = 0
        layers, bn_layers = [], []
        self.pool_features, self.scores_features = {}, {}
        
        for x in (cfg_features[self.architecture]):
            if isinstance(x, str) and x[:3] == 'avg':
                self.pool_features[str(layer)] = nn.AvgPool2d(kernel_size=self.kernel_size, stride=int(x[3:]), padding=padding)
                divisor *= 2
                continue
            elif isinstance(x, str) and x[:3] == 'max':
                self.pool_features[str(layer)] = nn.MaxPool2d(kernel_size=self.kernel_size, stride=int(x[3:]), padding=padding)
                continue
            elif isinstance(x, str) and x[:6] == 'atrous':
                layers += [nn.Conv2d(in_channels, int(x[6:]), kernel_size=self.kernel_size, padding=(self.kernel_size-1), stride=stride, dilation=dilation, bias=bias_flag)]
                in_channels = int(x[6:])
            elif isinstance(x, str) and x == 'score':
                self.scores_features[str(layer)] = nn.Conv2d(in_channels, self.dataset['num_cls'], kernel_size=1)
                self.scores_counter += 1
                continue
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=self.kernel_size, padding=padding, stride=stride, bias=bias_flag)]
                in_channels = x
            if self.bntt:
                bn_layers += [nn.ModuleList([nn.BatchNorm2d(in_channels, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.timesteps)])]
            layer += 1

        self.features = nn.ModuleList(layers)
        self.pool_features = nn.ModuleDict(self.pool_features)
        self.scores_features = nn.ModuleDict(self.scores_features)
        if self.bntt:
            self.bn_features = nn.ModuleList(bn_layers)
        
        layer = 0
        layers, bn_layers = [], []
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
                self.scores_counter += 1
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
            if self.bntt:
                bn_layers += [nn.ModuleList([nn.BatchNorm2d(in_channels, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.timesteps)])]
            layer += 1

        self.classifier = nn.ModuleList(layers)
        self.pool_classifier = nn.ModuleDict(self.pool_classifier)
        self.scores_classifier = nn.ModuleDict(self.scores_classifier)
        if self.bntt:
            self.bn_classifier = nn.ModuleList(bn_layers)

        if self.fcn:
            layers = []
            height = self.img_size[0]
            width = self.img_size[1]
            for _ in range(self.scores_counter):
                divisor = divisor//2
                layers += [nn.Sequential(
                    nn.ConvTranspose2d(self.dataset['num_cls'], self.dataset['num_cls'], kernel_size=self.kernel_size, stride=2, bias=False),
                    nn.UpsamplingBilinear2d((height//divisor, width//divisor))
                )]
            self.upsample = nn.ModuleList(layers)


    def _init_layers(self):
        if self.bntt:
            for bnlist in self.bn_features:
                for bn in bnlist:
                    bn.bias = None
            for bnlist in self.bn_classifier:
                for bn in bnlist:
                    bn.bias = None

        # Initialize the firing thresholds and weights of all the layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                if self.init == 'xavier':
                    torch.nn.init.xavier_uniform_(m.weight, gain=2)
                elif self.init == 'kaiming':
                    if self.ibt:
                        nn.init.kaiming_normal_(m.weight, a=self.alpha, mode='fan_in', nonlinearity='leaky_relu')
                    else:
                        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

                if m.bias is not None:
                    m.bias.data.zero_()

                m.threshold_pos = torch.tensor(self.def_threshold)
                m.threshold_neg = torch.tensor((-1/self.alpha)*self.def_threshold)


    def _initialize_mems(self, N, layer=None, get_layer_shape=False):

        height = self.img_size[0]
        width = self.img_size[1]

        # Initialize the neuronal membrane potentials
        layers = []
        divisor = 1
        i = 0
        for x in (cfg_features[self.architecture]):
            if isinstance(x, str) and ('avg' in x or 'max' in x):
                divisor *= 2
                continue
            elif isinstance(x, str) and x[:6] == 'atrous':
                x = int(x[6:])
            elif not isinstance(x, int):
                continue
            layers += [torch.zeros(N, x, height//divisor, width//divisor).cuda()] if self.gpu else [torch.zeros(N, x, height//divisor, width//divisor)]
            if get_layer_shape and i == layer:
                return layers[-1].shape
            i += 1
        mem_features = layers

        layers = []
        for x in (cfg_classifier[self.architecture]):
            if isinstance(x, str) and x == 'output':
                x = self.dataset['num_cls']
            elif isinstance(x, str) and ('avg' in x or 'max' in x):
                divisor *= 2
                continue
            elif isinstance(x, str) and x[:6] == 'atrous':
                x = int(x[6:])
            elif isinstance(x, str) and x != 'score':
                x = int(x.split('-')[0])
            else:
                continue
            layers += [torch.zeros(N, x, height//divisor, width//divisor).cuda()] if self.gpu else [torch.zeros(N, x, height//divisor, width//divisor)]
            if get_layer_shape and i == layer:
                return layers[-1].shape
            i += 1
        mem_classifier = layers

        if self.fcn:
            layers = []
            for _ in range(self.scores_counter):
                divisor = divisor//2
                if isinstance(x, int):
                    layers += [torch.zeros(N, self.dataset['num_cls'], height//divisor, width//divisor).cuda()] if self.gpu else [torch.zeros(N, self.dataset['num_cls'], height//divisor, width//divisor)]
                    if get_layer_shape and i == layer:
                        return layers[-1].shape
                    i += 1
            mem_upsample = layers

            return mem_features, mem_classifier, mem_upsample

        return mem_features, mem_classifier


    def _init_max_mem(self, N, layer, threshold_type):
        layer_shape = self._initialize_mems(N, layer, get_layer_shape=True)

        if threshold_type == 'layer':
            return 0.0
        elif threshold_type == 'channel':
            return torch.zeros(layer_shape[1]).cuda() if self.gpu else torch.zeros(layer_shape[1])
        elif threshold_type == 'neuron':
            return torch.zeros(layer_shape[1], layer_shape[2], layer_shape[3]).cuda() if self.gpu else torch.zeros(layer_shape[1], layer_shape[2], layer_shape[3])
        else:
            return None


    def threshold_update(self, scaling_factor=1.0, thresholds=[], threshold_type='layer'):
        # Initialize thresholds
        self.scaling_factor = scaling_factor
        
        for m in self.modules():
            if (isinstance(m, nn.Conv2d)) and thresholds:
                v_th = thresholds.pop(0)*self.scaling_factor
                if threshold_type == 'channel':
                    v_th = (v_th[:, None, None]).cuda() if self.gpu else (v_th[:, None, None])
                elif threshold_type == 'neuron':
                    v_th = (v_th[None, :, :, :]).cuda() if self.gpu else (v_th[None, :, :, :])
                m.threshold_pos = torch.tensor(v_th)
                m.threshold_neg = torch.tensor((-1/self.alpha)*v_th)


    def forward(self, x, find_max_mem=False, max_mem_layer=0):

        N, C, H, W = x[-1].size() if self.thl else x.size()

        if self.count_spikes:
            self.activations.updateCount(N)

        if self.fcn:
            mem_features, mem_classifier, mem_upsample = self._initialize_mems(N)
        else:
            mem_features, mem_classifier = self._initialize_mems(N)

        spikes = []

        max_mem = self._init_max_mem(N, max_mem_layer, find_max_mem)

        total_timesteps = (self.timesteps + len(mem_features) + len(mem_classifier)) if self.full_prop else self.timesteps

        for t in range(total_timesteps):
            if t < self.timesteps:
                out_prev = x[t] if self.thl else self.input_layer(x)
            elif self.gpu:
                out_prev = torch.zeros_like(x[-1]).cuda() if self.thl else torch.zeros_like(x).cuda()
            else:
                out_prev = torch.zeros_like(x[-1]) if self.thl else torch.zeros_like(x)

            for k in range(len(self.features)):

                if find_max_mem and k == max_mem_layer:
                    if find_max_mem == 'layer':
                        ts_max = (self.features[k](out_prev)).max()
                        max_mem = ts_max if ts_max > max_mem else max_mem
                    elif find_max_mem == 'channel':
                        out = self.features[k](out_prev)
                        out = out.permute(1, 0, 2, 3).reshape(out.size(1), -1)
                        ts_max = torch.max(out, 1).values
                        max_mem = ((ts_max > max_mem) * ts_max) + ((ts_max < max_mem) * max_mem)
                    elif find_max_mem == 'neuron':
                        out = self.features[k](out_prev)
                        out = out.permute(1, 2, 3, 0)
                        ts_max = torch.max(out, 3).values
                        max_mem = ((ts_max > max_mem) * ts_max) + ((ts_max < max_mem) * max_mem)
                    break

                if ((not self.bntt) or (self.full_prop and t >= self.timesteps)):
                    mem_features[k] = (self.leak_mem * mem_features[k] + (self.features[k](out_prev)))
                else:
                    mem_features[k] = (self.leak_mem * mem_features[k] + (self.bn_features[k][t](self.features[k](out_prev))))
                mem_thr_pos         = (mem_features[k]/self.features[k].threshold_pos) - 1.0
                if self.ibt:
                    mem_thr_neg 	= (mem_features/self.features[k].threshold_neg) - 1.0
                else:
                    mem_thr_neg     = torch.zeros_like(mem_thr_pos)
                out                 = self.spike_fn(mem_thr_pos, self.gpu)
                rst                 = torch.zeros_like(mem_features[k]).cuda() if self.gpu else torch.zeros_like(mem_features[k])
                rst                 = (mem_thr_pos > 0) * self.features[k].threshold_pos + (mem_thr_neg > 0) * self.features[k].threshold_neg
                mem_features[k]     = mem_features[k] - rst
                out_prev            = out.clone()

                if self.count_spikes:
                    self.activations.updateSum(k, torch.sum(out.detach().clone()).item())

                if (self.fcn) and (str(k+1) in self.scores_features.keys()):
                    spikes.insert(0, self.scores_features[str(k+1)](out_prev))

                if str(k+1) in self.pool_features.keys():
                    out = self.pool_features[str(k+1)](out_prev)
                    out_prev = out.clone()

            if find_max_mem and max_mem_layer < len(self.features):
                continue

            prev = len(self.features)

            for k in range(len(self.classifier) if self.fcn else (len(self.classifier) - 1)):
                
                if find_max_mem and (prev + k) == max_mem_layer:
                    if find_max_mem == 'layer':
                        ts_max = (self.classifier[k](out_prev)).max()
                        max_mem = ts_max if ts_max > max_mem else max_mem
                    elif find_max_mem == 'channel':
                        out = self.classifier[k](out_prev)
                        out = out.permute(1, 0, 2, 3).reshape(out.size(1), -1)
                        ts_max = torch.max(out, 1).values
                        max_mem = ((ts_max > max_mem) * ts_max) + ((ts_max < max_mem) * max_mem)
                    elif find_max_mem == 'neuron':
                        out = self.classifier[k](out_prev)
                        out = out.permute(1, 2, 3, 0)
                        ts_max = torch.max(out, 3).values
                        max_mem = ((ts_max > max_mem) * ts_max) + ((ts_max < max_mem) * max_mem)
                    break

                if ((not self.bntt) or (self.full_prop and t >= self.timesteps)):
                    mem_classifier[k] = (self.leak_mem * mem_classifier[k] + (self.classifier[k](out_prev)))
                else:
                    mem_classifier[k] = (self.leak_mem * mem_classifier[k] + (self.bn_classifier[k][t](self.classifier[k](out_prev))))
                mem_thr_pos         = (mem_classifier[k]/self.classifier[k].threshold_pos) - 1.0
                if self.ibt:
                    mem_thr_neg 	= (mem_classifier/self.classifier[k].threshold_neg) - 1.0
                else:
                    mem_thr_neg     = torch.zeros_like(mem_thr_pos)
                out                 = self.spike_fn(mem_thr_pos, self.gpu)
                rst                 = torch.zeros_like(mem_classifier[k]).cuda() if self.gpu else torch.zeros_like(mem_classifier[k])
                rst                 = (mem_thr_pos > 0) * self.classifier[k].threshold_pos + (mem_thr_neg > 0) * self.classifier[k].threshold_neg
                mem_classifier[k]   = mem_classifier[k] - rst
                out_prev            = out.clone()

                if self.count_spikes:
                    self.activations.updateSum((prev+k), torch.sum(out.detach().clone()).item())
                
                if (self.fcn) and (str(k+1) in self.scores_classifier.keys()):
                    spikes.insert(0, self.scores_classifier[str(k+1)](out_prev))

                if str(k+1) in self.pool_classifier.keys():
                    out = self.pool_classifier[str(k+1)](out_prev)
                    out_prev = out.clone()

            if self.fcn:
                prev += len(self.classifier)

                out_prev = spikes[0]

                for k in range(len(self.upsample) - 1):

                    mem_upsample[k]     = (self.leak_mem * mem_upsample[k] + (self.upsample[k](out_prev)) + spikes[k+1])
                    mem_thr_pos         = (mem_upsample[k]/self.upsample[k].threshold_pos) - 1.0
                    if self.ibt:
                        mem_thr_neg 	= (mem_upsample/self.upsample[k].threshold_neg) - 1.0
                    else:
                        mem_thr_neg     = torch.zeros_like(mem_thr_pos)
                    out                 = self.spike_fn(mem_thr_pos, self.gpu)
                    rst                 = torch.zeros_like(mem_upsample[k]).cuda() if self.gpu else torch.zeros_like(mem_upsample[k])
                    rst                 = (mem_thr_pos > 0) * self.upsample[k].threshold_pos + (mem_thr_neg > 0) * self.upsample[k].threshold_neg
                    mem_upsample[k]     = mem_upsample[k] - rst
                    out_prev            = out.clone()

                    if self.count_spikes:
                        self.activations.updateSum((prev+k), torch.sum(out.detach().clone()).item())

            # compute last conv
            if not find_max_mem:
                if self.fcn:
                    mem_upsample[k+1] = (1 * mem_upsample[k+1] + (self.upsample[k+1](out_prev)) + spikes[k+2])
                else:
                    mem_classifier[k+1] = (1 * mem_classifier[k+1] + self.classifier[k+1](out_prev))

        if find_max_mem:
            return max_mem

        if self.fcn:
            out_voltage = mem_upsample[k+1]
        else:
            out_voltage = mem_classifier[k+1]
            out_voltage = (out_voltage) / self.timesteps
            out_voltage = F.interpolate(out_voltage, (H, W), mode='bilinear', align_corners=True)

        return out_voltage