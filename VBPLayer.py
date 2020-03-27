import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
import math
import numpy as np


class VBPLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_prec=10, isoutput=False):
        super(VBPLinear, self).__init__()
        self.n_in = in_features
        self.n_out = out_features

        self.prior_prec = prior_prec
        self.isoutput = isoutput


        self.bias = nn.Parameter(th.Tensor(out_features))
        self.mu_w = nn.Parameter(th.Tensor(out_features, in_features))
        self.logsig2_w = nn.Parameter(th.Tensor(out_features, in_features))
        self.reset_parameters()

        self.normal = False
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.mu_w.size(1))
        self.mu_w.data.normal_(0, stdv)
        self.logsig2_w.data.zero_().normal_(-9, 0.001)
        self.bias.data.zero_()

    def forward(self, data):
        return F.linear(data, self.mu_w, self.bias)

    def KL(self, loguniform=False):
        if loguniform:
            k1 = 0.63576; k2 = 1.87320; k3 = 1.48695
            log_alpha = self.logsig2_w - 2 * th.log(self.mu_w.abs() + 1e-8)
            kl = -th.sum(k1 * F.sigmoid(k2 + k3 * log_alpha) - 0.5 * F.softplus(-log_alpha) - k1)
        else:
            logsig2_w = self.logsig2_w.clamp(-11, 11)
            kl = 0.5 * (self.prior_prec * (self.mu_w.pow(2) + logsig2_w.exp()) - logsig2_w - 1 - np.log(self.prior_prec)).sum()
        return kl

    def var(self, prev_mean, prev_var=None, C=100):
        if self.isoutput:
            m2s2_w = self.mu_w.pow(2) + self.logsig2_w.exp()
            term1 = F.linear(prev_var, m2s2_w)
            term2 = F.linear(prev_mean.pow(2), self.logsig2_w.exp())
            return term1 + term2 

        else:
            pZ = th.sigmoid(C * F.linear(prev_mean, self.mu_w, self.bias))

            # Compute var[h]
            if prev_var is None:
                term1 = 0
            else:
                m2s2_w = self.mu_w.pow(2) + self.logsig2_w.exp()
                term1 = F.linear(prev_var, m2s2_w)
            term2 = F.linear(prev_mean.pow(2), self.logsig2_w.exp())
            varh = term1 + term2 

            # Compute E[h]^2
            term3 = F.linear(prev_mean, self.mu_w, self.bias).pow(2)

            return pZ * varh + pZ * (1 - pZ) * term3

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.n_in) + ' -> ' \
               + str(self.n_out) \
               + f', isoutput={self.isoutput})'


class VBPConv(VBPLinear):
    def __init__(self, in_channels, out_channels, kernel_size, prior_prec=10, stride=1,
                 padding=0, dilation=1, groups=1, isoutput=False):
        super(VBPLinear, self).__init__()
        self.n_in = in_channels
        self.n_out = out_channels

        self.kernel_size = kernel_size
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups

        self.prior_prec = prior_prec
        self.isoutput = isoutput

        self.bias = nn.Parameter(th.Tensor(out_channels))
        self.mu_w = nn.Parameter(th.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        self.logsig2_w = nn.Parameter(th.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        self.reset_parameters()


    def reset_parameters(self):
        n = self.n_in
        for k in range(1, self.kernel_size):
            n *= k
        self.mu_w.data.normal_(0, 1. / math.sqrt(n))
        self.logsig2_w.data.zero_().normal_(-9, 0.001)
        self.bias.data.zero_()

    def forward(self, data):
        return F.conv2d(data, self.mu_w, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def var(self, prev_mean, prev_var=None, C=100):
        if self.isoutput:
            m2s2_w = self.mu_w.pow(2) + self.logsig2_w.exp()
            term1 = F.conv2d(prev_var, m2s2_w, None, self.stride, self.padding, self.dilation, self.groups)
            term2 = F.conv2d(prev_mean.pow(2), self.logsig2_w.exp(), None, self.stride, self.padding, self.dilation, self.groups)
            return term1 + term2 

        else:
            pZ = th.sigmoid(C*F.conv2d(prev_mean, self.mu_w, self.bias, self.stride, self.padding, self.dilation, self.groups))

            # Compute var[h]
            if prev_var is None:
                term1 = 0
            else:
                m2s2_w = self.mu_w.pow(2) + self.logsig2_w.exp()
                term1 = F.conv2d(prev_var, m2s2_w, None, self.stride, self.padding, self.dilation, self.groups)
            term2 = F.conv2d(prev_mean.pow(2), self.logsig2_w.exp(), None, self.stride, self.padding, self.dilation, self.groups)
            varh = term1 + term2 


            # Compute E[h]^2
            term3 = F.conv2d(prev_mean, self.mu_w, self.bias, self.stride, self.padding, self.dilation, self.groups).pow(2)

            return pZ * varh + pZ * (1 - pZ) * term3

    def __repr__(self):
        s = ('{name}({n_in}, {n_out}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        s += ', bias=True'
        s += ')'

        return s.format(name=self.__class__.__name__, **self.__dict__)


