"""Contains novel layer definitions."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter
import pdb

DEFAULT_THRESHOLD = 0.5#0.5#5e-3

class ThresholdSTE(torch.autograd.Function):
    """Threshold {0, x} a real valued tensor."""

    def __init__(self, threshold=DEFAULT_THRESHOLD):
        super(ThresholdSTE, self).__init__()
        self.threshold = threshold

    def forward(self, inputs):
        outputs = inputs.clone()
        idx = inputs.gt(self.threshold)
        outputs[idx] = inputs[idx]
        outputs[1-idx] = 0

        return outputs

    def backward(self, gradOutput):
        return gradOutput


class Threshold(torch.autograd.Function):
    """Threshold {0, x} a real valued tensor."""

    @staticmethod
    def forward(ctx, threshold, inputs):
        outputs = inputs.clone()
        idx = inputs.gt(threshold)
        outputs[idx] = inputs[idx]
        outputs[1-idx] = 0
        ctx.save_for_backward(outputs)
        ctx.thresh = threshold

        return outputs#inputs.clamp(min=self.threshold)

    @staticmethod
    def backward(ctx, gradOutput):
        outputs, = ctx.saved_tensors
        threshold = ctx.thresh
        gradInput = gradOutput.clone()
        gradInput[outputs.le(threshold)] = 0
        return None,gradInput

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class BinarizerSTEInstance(torch.autograd.Function):
    """Binarizes {0, 1} a real valued tensor. Backward is STE"""

    def __init__(self, threshold=DEFAULT_THRESHOLD):
        super(BinarizerSTEInstance, self).__init__()
        #self.threshold = torch.FloatTensor(size=[1]).fill_(threshold)
        #self.threshold = Parameter(self.threshold, requires_grad = False)#True
        self.threshold = threshold

    def forward(self, inputs):
        outputs = inputs.clone()
        outputs[inputs.le(self.threshold)] = 0
        outputs[inputs.gt(self.threshold)] = 1
        return outputs

    def backward(self, gradOutput):
        return gradOutput

class BinarizerSTEStatic(torch.autograd.Function):
    """Binarizes {0, 1} a real valued tensor. Backward is STE"""

    @staticmethod
    def forward(ctx, threshold, inputs):
        outputs = inputs.clone()
        outputs[inputs.le(threshold)] = 0
        outputs[inputs.gt(threshold)] = 1

        return outputs

    @staticmethod
    def backward(ctx, gradOutput):
        gradInput = gradOutput.clone()

        return None,gradInput


class Binarizer(torch.autograd.Function):
    """Binarizes {0, 1} a real valued tensor. Backward is ReLU like"""

    @staticmethod
    def forward(ctx, threshold, inputs):
        outputs = inputs.clone()
        outputs[inputs.le(threshold)] = 0
        outputs[inputs.gt(threshold)] = 1
        ctx.save_for_backward(outputs)

        return outputs

    @staticmethod
    def backward(ctx, gradOutput):
        outputs, = ctx.saved_tensors
        gradInput = gradOutput.clone() * outputs #cancel gradient from pruned filters

        return None,gradInput


class BinarizerP(torch.autograd.Function):
    """Binarizes {0, x} a real valued tensor with learnable threshold."""

    def __init__(self, threshold=DEFAULT_THRESHOLD):
        super(BinarizerP, self).__init__()
        self.threshold = torch.FloatTensor(size=[1]).fill_(threshold)
        self.threshold = Parameter(self.threshold, requires_grad = False)#True

    def forward(self, inputs):
        outputs = inputs.clone()
        outputs[inputs.lt(self.threshold)] = 0
        outputs[inputs.ge(self.threshold)] = 1
        '''
        outputs = torch.nn.Threshold(self.threshold,0.0)
        outputs = outputs/inputs
        '''
        return outputs

    def backward(self, gradOutput):
        return gradOutput


class Ternarizer(torch.autograd.Function):
    """Ternarizes {-1, 0, 1} a real valued tensor."""

    def __init__(self, threshold=DEFAULT_THRESHOLD):
        super(Ternarizer, self).__init__()
        self.threshold = threshold

    def forward(self, inputs):
        outputs = inputs.clone()
        outputs.fill_(0)
        outputs[inputs < 0] = -1
        outputs[inputs > self.threshold] = 1
        return outputs

    def backward(self, gradOutput):
        return gradOutput

class HyperNetwork(nn.Module):

    def __init__(self, f_size = 1, z_dim = 64, out_size=16, in_size=1):
        super(HyperNetwork, self).__init__()
        self.z_dim = z_dim
        self.f_size = f_size
        self.out_size = out_size
        self.in_size = in_size

        self.w1 = Parameter(torch.ones((self.z_dim, self.out_size*self.f_size*self.f_size))*1e-2)
        self.b1 = Parameter(torch.zeros((self.out_size*self.f_size*self.f_size)))

        self.w2 = Parameter(torch.ones((self.z_dim, self.in_size*self.z_dim))*1e-2)
        self.b2 = Parameter(torch.zeros((self.in_size*self.z_dim)))

    def forward(self, z):

        h_in = torch.matmul(z, self.w2) + self.b2
        h_in = h_in.view(self.in_size, self.z_dim)

        h_final = torch.matmul(h_in, self.w1) + self.b1
        kernel = h_final.view(self.out_size, self.in_size, self.f_size, self.f_size)

        return kernel

class MaskSTE(nn.Module):
    """Applies differentiable mask on the input."""

    def __init__(self, out_channels, mask_init='1s', mask_scale=1e-2,
                 threshold_fn='binarizer', threshold=None, mask_factor=1, applysig=True, layertype='conv', hypernet=False):
        super(MaskSTE, self).__init__()
        self.mask_scale = mask_scale
        self.mask_init = mask_init
        #self.mask_factor = mask_factor
        #self.mask_real_post = nn.Sigmoid() if applysig else Identity()
        self.mask_factor = torch.FloatTensor([mask_factor])
        self.mask_factor = Parameter(self.mask_factor, requires_grad = False)
        self.applysig = applysig

        if threshold is None:
            threshold = DEFAULT_THRESHOLD if applysig else 1e-4
        self.info = {
            'threshold_fn': threshold_fn,
            'threshold': threshold,
        }

        self.out_channels = out_channels
        self.hypernet_mask = None

        if hypernet:
            z_dim = 64
            self.hypernet_mask = HyperNetwork(out_size = out_channels, z_dim=z_dim)
            self.z_mask = torch.FloatTensor(size=[z_dim])
            self.z_mask.fill_(1)
            #self.z_mask = torch.randn(z_dim)

        if hypernet:
            #TODO don't handle layetype yet
            z_dim = 64
            self.hypernet_mask = HyperNetwork(out_size = out_channels, z_dim=z_dim)
            self.z_mask = torch.FloatTensor(size=[z_dim])
            self.z_mask.fill_(1)
            #self.z_mask = torch.randn(z_dim)

            self.z_mask = Parameter(self.z_mask, requires_grad = False)
        else:
            # Initialize real-valued mask weights.
            if layertype == 'linear':
                self.mask_real = torch.FloatTensor(size=[1,out_channels])
            else:
                self.mask_real = torch.FloatTensor(size=[1,out_channels,1,1])

            if mask_init == 'uniform':
                self.mask_real.uniform_(-1 * mask_scale, mask_scale)
            elif mask_init == '1s':
                self.mask_real.fill_(mask_scale)
            # mask_real is now a trainable parameter.
            self.mask_real = Parameter(self.mask_real, requires_grad = True)

        # Initialize the thresholder.
        if threshold_fn == 'binarizer':
            #print('Calling binarizer with threshold:', threshold)
            self.threshold_fn = BinarizerSTEInstance(threshold=threshold)
            #self.threshold_fn = BinarizerSTEStatic.apply#(threshold=threshold)
            self.thresholdp = threshold
        elif threshold_fn == 'binarizerStatic':
            self.threshold_fn = BinarizerSTEStatic.apply#(threshold=threshold)
            self.thresholdp = threshold
        elif threshold_fn == 'binarizerNoSTE':
            #print('Calling binarizer with threshold:', threshold)
            self.threshold_fn = Binarizer.apply
            self.thresholdp = threshold
        elif threshold_fn == 'ternarizer':
            print('Calling ternarizer with threshold:', threshold)
            self.threshold_fn = Ternarizer(threshold=threshold)
        elif threshold_fn == 'binarizerp':
            self.threshold_fn = BinarizerP(threshold=threshold)
            self.thresholdp = self.threshold_fn.threshold
        elif threshold_fn == 'threshold':
            self.threshold_fn = ThresholdSTE(threshold=threshold)
            self.thresholdp = threshold
        elif threshold_fn == 'thresholdNoSTE':
            #self.threshold_fn = ThresholdSTE(threshold=threshold)
            self.threshold_fn = Threshold.apply
            self.thresholdp = threshold



    def forward(self, inputfeat):
        postprocess = nn.Sigmoid() if self.applysig else Identity()
        if self.hypernet_mask is not None:
            self.mask_real = self.hypernet_mask(self.z_mask).view(-1)
        mask_real_sig = postprocess(self.mask_real)

        #TODO needs different check as threshold_fn is a class not function
        #Functions are used only once, so create the instance in forward but the pointer to class in init
        #https://github.com/pytorch/pytorch/issues/821
        if 'apply' in str(self.threshold_fn):
            mask_thresholded = self.threshold_fn(self.thresholdp, mask_real_sig)
        else:
            mask_thresholded = self.threshold_fn(mask_real_sig)

        res = inputfeat * mask_thresholded
        return res

    def get_binary_mask(self):
        postprocess = nn.Sigmoid() if self.applysig else Identity()
        if self.hypernet_mask is not None:
            self.mask_real = self.hypernet_mask(self.z_mask).view(-1)
        mask_real_sig = postprocess(self.mask_real)

        #TODO needs different check as threshold_fn is a class not function
        #Functions are used only once, so create the instance in forward but the pointer to class in init
        #https://github.com/pytorch/pytorch/issues/821
        if 'apply' in str(self.threshold_fn):
            mask_thresholded = self.threshold_fn(self.thresholdp, mask_real_sig)
        else:
            mask_thresholded = self.threshold_fn(mask_real_sig)
        mask_1d = mask_thresholded.view(-1)

        return mask_1d, self.mask_factor


    def __repr__(self):
        s = ('{name} ({out_channels}, {threshold_fn}, {mask_init}, mask_scale={mask_scale})')
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def _apply(self, fn):
        for module in self.children():
            module._apply(fn)

        for param in self._parameters.values():
            if param is not None:
                # Variables stored in modules are graph leaves, and we don't
                # want to create copy nodes, so we have to unpack the data.
                param.data = fn(param.data)
                if param._grad is not None:
                    param._grad.data = fn(param._grad.data)

        for key, buf in self._buffers.items():
            if buf is not None:
                self._buffers[key] = fn(buf)

