#---------------------------------------------------
# Imports
#---------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys


# --------------------------------------------------
# Poisson spike generator
#   Positive spike is generated (i.e.  1 is returned) if rand()<=abs(input) and sign(input)= 1
#   Negative spike is generated (i.e. -1 is returned) if rand()<=abs(input) and sign(input)=-1
# --------------------------------------------------
class PoissonGenerator(nn.Module):
    
    def __init__(self, gpu=False):
        super().__init__()

        self.gpu = gpu

    def forward(self, inp, rescale_fac=1.0):
        rand_inp = torch.rand_like(inp).cuda() if self.gpu else torch.rand_like(inp)
        return torch.mul(torch.le(rand_inp * rescale_fac, torch.abs(inp)).float(), torch.sign(inp))


# --------------------------------------------------
# Spiking neuron with fast-sigmoid surrogate gradient
# This class is replicated from:
# https://github.com/fzenke/spytorch/blob/master/notebooks/SpyTorchTutorial2.ipynb
# --------------------------------------------------
class SuperSpike(torch.autograd.Function):
    """
    Here we implement our spiking nonlinearity which also implements
    the surrogate gradient. By subclassing torch.autograd.Function,
    we will be able to use all of PyTorch's autograd functionality.
    Here we use the normalized negative part of a fast sigmoid as
    was done in Zenke & Ganguli (2018).
    """
    scale = 100.0  # Controls the steepness of the fast-sigmoid surrogate gradient

    @staticmethod
    def forward(ctx, input, gpu):
        """
        In the forward pass, we compute a step function of the input Tensor and
        return it. ctx is a context object that we use to stash information which
        we need to later backpropagate our error signals. To achieve this we use
        the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        out = torch.zeros_like(input).cuda() if gpu else torch.zeros_like(input)
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass, we receive a Tensor we need to compute
        the surrogate gradient of the loss with respect to the input.
        Here we use the normalized negative part of a fast sigmoid
        as was done in Zenke & Ganguli (2018).
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input / (SuperSpike.scale * torch.abs(input) + 1.0) ** 2
        return grad


# --------------------------------------------------
# Spiking neuron with piecewise-linear surrogate gradient
# --------------------------------------------------
class LinearSpike(torch.autograd.Function):
    """
    Here we implement our spiking nonlinearity which also implements
    the surrogate gradient. By subclassing torch.autograd.Function,
    we will be able to use all of PyTorch's autograd functionality.
    Here we use the piecewise-linear surrogate gradient as was done
    in Bellec et al. (2018).
    """
    gamma = 0.3  # Controls the dampening of the piecewise-linear surrogate gradient

    @staticmethod
    def forward(ctx, input, gpu):
        """
        In the forward pass, we compute a step function of the input Tensor and
        return it. ctx is a context object that we use to stash information which
        we need to later backpropagate our error signals. To achieve this we use
        the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        out = torch.zeros_like(input).cuda() if gpu else torch.zeros_like(input)
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass, we receive a Tensor we need to compute
        the surrogate gradient of the loss with respect to the input.
        Here we use the piecewise-linear surrogate gradient as was
        done in Bellec et al. (2018).
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input * LinearSpike.gamma * F.threshold(1.0 - torch.abs(input), 0, 0)
        return grad


# --------------------------------------------------
# Spiking neuron with exponential surrogate gradient
# --------------------------------------------------
class ExpSpike(torch.autograd.Function):
    """
    Here we implement our spiking nonlinearity which also implements
    the surrogate gradient. By subclassing torch.autograd.Function,
    we will be able to use all of PyTorch's autograd functionality.
    Here we use the exponential surrogate gradient as was done in
    Shrestha et al. (2018).
    """
    alpha = 1.0  # Controls the magnitude of the exponential surrogate gradient
    beta = 10.0  # Controls the steepness of the exponential surrogate gradient

    @staticmethod
    def forward(ctx, input, gpu):
        """
        In the forward pass, we compute a step function of the input Tensor and
        return it. ctx is a context object that we use to stash information which
        we need to later backpropagate our error signals. To achieve this we use
        the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        out = torch.zeros_like(input).cuda() if gpu else torch.zeros_like(input)
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass, we receive a Tensor we need to compute
        the surrogate gradient of the loss with respect to the input.
        Here we use the exponential surrogate gradient as was done
        in Shrestha et al. (2018).
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input * ExpSpike.alpha * torch.exp(-ExpSpike.beta * torch.abs(input))
        return grad


# --------------------------------------------------
# Spiking neuron with pass-through surrogate gradient
# --------------------------------------------------
class PassThruSpike(torch.autograd.Function):
    """
    Here we implement our spiking nonlinearity which also implements
    the surrogate gradient. By subclassing torch.autograd.Function,
    we will be able to use all of PyTorch's autograd functionality.
    Here we use the pass-through surrogate gradient.
    """

    @staticmethod
    def forward(ctx, input, gpu):
        """
        In the forward pass, we compute a step function of the input Tensor and
        return it. For this spiking nonlinearity, the context object ctx does not
        stash input information since it is not used for backpropagation.
        """
        # ctx.save_for_backward(input)
        out = torch.zeros_like(input).cuda() if gpu else torch.zeros_like(input)
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass, we receive a Tensor we need to compute
        the surrogate gradient of the loss with respect to the input.
        Here we use the pass-through surrogate gradient.
        """
        # input,   = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input


# Overwrite the naive spike function by differentiable spiking nonlinearity which implements a surrogate gradient
def init_spike_fn(grad_type):
    if (grad_type == 'FastSigm'):
        spike_fn = SuperSpike.apply
    elif (grad_type == 'Linear'):
        spike_fn = LinearSpike.apply
    elif (grad_type == 'Exp'):
        spike_fn = ExpSpike.apply
    elif (grad_type == 'PassThru'):
        spike_fn = PassThruSpike.apply
    else:
        sys.exit("Unknown gradient type '{}'".format(grad_type))
    return spike_fn