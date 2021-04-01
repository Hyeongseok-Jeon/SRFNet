# Copyright (c) 2020 Uber Technologies, Inc.
# Please check LICENSE for more detail


import numpy as np
from fractions import gcd
from numbers import Number

import torch
from torch import nn
from torch.nn import functional as F


class GAT_SRF(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, training, concat=True):
        super(GAT_SRF, self).__init__()
        self.training = training
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(2 * out_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, out_features)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, x):
        h = x[0]
        adj = x[1]
        Wh = torch.mm(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        a_input = self._prepare_attentional_mechanism_input(adj)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention_raw = F.softmax(attention, dim=1)
        attention = F.dropout(attention_raw, self.dropout, training=self.training)
        h_mod = torch.matmul(torch.transpose(attention, 1, 2), Wh)
        h_prime = torch.diagonal(h_mod, 0, 1, 2)

        if self.concat:
            return [F.elu(h_prime), attention_raw]
        else:
            return [h_prime, attention_raw]

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0]  # number of nodes

        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)

        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        # all_combinations_matrix.shape == (N * N, 2 * out_features)

        return all_combinations_matrix.view(N, N, 2 * self.out_features)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, training, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.training = training
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.a = nn.Parameter(torch.empty(size=(2 * out_features, out_features)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, x, share_weight):
        h = x[0]
        adj = x[1]
        Wh = torch.mm(h, share_weight)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention_raw = F.softmax(attention, dim=1)
        attention = F.dropout(attention_raw, self.dropout, training=self.training)
        h_mod = torch.matmul(torch.transpose(attention, 1, 2), Wh)
        h_prime = torch.diagonal(h_mod, 0, 1, 2)

        if self.concat:
            return [F.elu(h_prime), attention_raw]
        else:
            return [h_prime, attention_raw]

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0]  # number of nodes

        # Below, two matrices are created that contain embeddings in their rows in different orders.
        # (e stands for embedding)
        # These are the rows of the first matrix (Wh_repeated_in_chunks):
        # e1, e1, ..., e1,            e2, e2, ..., e2,            ..., eN, eN, ..., eN
        # '-------------' -> N times  '-------------' -> N times       '-------------' -> N times
        #
        # These are the rows of the second matrix (Wh_repeated_alternating):
        # e1, e2, ..., eN, e1, e2, ..., eN, ..., e1, e2, ..., eN
        # '----------------------------------------------------' -> N times
        #

        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        # Wh_repeated_in_chunks.shape == Wh_repeated_alternating.shape == (N * N, out_features)

        # The all_combination_matrix, created below, will look like this (|| denotes concatenation):
        # e1 || e1
        # e1 || e2
        # e1 || e3
        # ...
        # e1 || eN
        # e2 || e1
        # e2 || e2
        # e2 || e3
        # ...
        # e2 || eN
        # ...
        # eN || e1
        # eN || e2
        # eN || e3
        # ...
        # eN || eN

        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        # all_combinations_matrix.shape == (N * N, 2 * out_features)

        return all_combinations_matrix.view(N, N, 2 * self.out_features)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class GraphAttentionLayer_time_serial(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, config, in_features, out_features, dropout, alpha, training, nTime, concat=True):
        super(GraphAttentionLayer_time_serial, self).__init__()
        self.training = training
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.nTime = nTime
        self.config = config
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, out_features)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, x, share_weight):
        h = x[0]
        adj = x[1]
        vehicle_num = int(h.shape[0]/self.nTime)
        Wh = torch.mm(h, share_weight)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15 * torch.ones_like(adj)
        # attention = torch.where(adj > 0, e, zero_vec)
        attention = [torch.where(adj > 0, e[vehicle_num*i:vehicle_num*(i+1), vehicle_num*i:vehicle_num*(i+1), :], zero_vec) for i in range(self.nTime)]
        attention_raw = [F.softmax(attention[i], dim=1) for i in range(self.nTime)]
        attention = [F.dropout(attention_raw[i], self.dropout, training=self.training) for i in range(self.nTime)]
        h_mod = [torch.matmul(torch.transpose(attention[i], 1, 2), Wh[vehicle_num*i:vehicle_num*(i+1), :]) for i in range(self.nTime)]
        h_prime = [torch.diagonal(h_mod[i], 0, 1, 2) for i in range(self.nTime)]

        if self.concat:
            return [F.elu(h_prime[i]) for i in range(self.nTime)], attention_raw
        else:
            return h_prime, attention_raw

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0]  # number of nodes

        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)

        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        # all_combinations_matrix.shape == (N * N, 2 * out_features)

        return all_combinations_matrix.view(N, N, 2 * self.out_features)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""

    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(1, 2 * out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, input, adj):
        dv = 'cuda' if input.is_cuda else 'cpu'

        N = input.size()[0]
        edge = adj.nonzero().t()

        h = torch.mm(input, self.W)
        # h: N x out
        assert not torch.isnan(h).any()

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # edge: 2*D x E

        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N, 1), device=dv))
        # e_rowsum: N x 1

        edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out

        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


# Conv layer with norm (gn or bn) and relu. 
class Conv(nn.Module):
    def __init__(self, n_in, n_out, kernel_size=3, stride=1, norm='GN', ng=32, act=True):
        super(Conv, self).__init__()
        assert(norm in ['GN', 'BN', 'SyncBN'])

        self.conv = nn.Conv2d(n_in, n_out, kernel_size=kernel_size, padding=(int(kernel_size) - 1) // 2, stride=stride, bias=False)
        
        if norm == 'GN':
            self.norm = nn.GroupNorm(gcd(ng, n_out), n_out)
        elif norm == 'BN':
            self.norm = nn.BatchNorm2d(n_out)
        else:
            exit('SyncBN has not been added!')

        self.relu = nn.ReLU(inplace=True)
        self.act = act    

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        if self.act:
            out = self.relu(out)
        return out


class Conv1d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size=3, stride=1, norm='GN', ng=32, act=True):
        super(Conv1d, self).__init__()
        assert(norm in ['GN', 'BN', 'SyncBN'])

        self.conv = nn.Conv1d(n_in, n_out, kernel_size=kernel_size, padding=(int(kernel_size) - 1) // 2, stride=stride, bias=False)

        if norm == 'GN':
            self.norm = nn.GroupNorm(gcd(ng, n_out), n_out)
        elif norm == 'BN':
            self.norm = nn.BatchNorm1d(n_out)
        else:
            exit('SyncBN has not been added!')

        self.relu = nn.ReLU(inplace=True)
        self.act = act

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        if self.act:
            out = self.relu(out)
        return out


class Linear(nn.Module):
    def __init__(self, n_in, n_out, norm='GN', ng=32, act=True):
        super(Linear, self).__init__()
        assert(norm in ['GN', 'BN', 'SyncBN'])

        self.linear = nn.Linear(n_in, n_out, bias=False)
        
        if norm == 'GN':
            self.norm = nn.GroupNorm(gcd(ng, n_out), n_out)
        elif norm == 'BN':
            self.norm = nn.BatchNorm1d(n_out)
        else:
            exit('SyncBN has not been added!')
        
        self.relu = nn.ReLU(inplace=True)
        self.act = act

    def forward(self, x):
        out = self.linear(x)
        out = self.norm(out)
        if self.act:
            out = self.relu(out)
        return out


# Post residual layer
class PostRes(nn.Module):
    def __init__(self, n_in, n_out, stride=1, norm='GN', ng=32, act=True):
        super(PostRes, self).__init__()
        assert(norm in ['GN', 'BN', 'SyncBN'])
        
        self.conv1 = nn.Conv2d(n_in, n_out, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(n_out, n_out, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace = True)
        
        # All use name bn1 and bn2 to load imagenet pretrained weights
        if norm == 'GN':
            self.bn1 = nn.GroupNorm(gcd(ng, n_out), n_out)
            self.bn2 = nn.GroupNorm(gcd(ng, n_out), n_out)
        elif norm == 'BN':
            self.bn1 = nn.BatchNorm2d(n_out)
            self.bn2 = nn.BatchNorm2d(n_out)
        else:
            exit('SyncBN has not been added!')

        if stride != 1 or n_out != n_in:
            if norm == 'GN':
                self.downsample = nn.Sequential(
                        nn.Conv2d(n_in, n_out, kernel_size=1, stride=stride, bias=False),
                        nn.GroupNorm(gcd(ng, n_out), n_out))
            elif norm == 'BN':
                self.downsample = nn.Sequential(
                        nn.Conv2d(n_in, n_out, kernel_size=1, stride=stride, bias=False),
                        nn.BatchNorm2d(n_out))
            else:
                exit('SyncBN has not been added!')    
        else:
            self.downsample = None

        self.act = act

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x
        if self.act:
            out = self.relu(out)
        return out


class Res1d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size=3, stride=1, norm='GN', ng=32, act=True):
        super(Res1d, self).__init__()
        assert(norm in ['GN', 'BN', 'SyncBN'])
        padding = (int(kernel_size) - 1) // 2
        self.conv1 = nn.Conv1d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.conv2 = nn.Conv1d(n_out, n_out, kernel_size=kernel_size, padding=padding, bias=False)
        self.relu = nn.ReLU(inplace = True)

        # All use name bn1 and bn2 to load imagenet pretrained weights
        if norm == 'GN':
            self.bn1 = nn.GroupNorm(gcd(ng, n_out), n_out)
            self.bn2 = nn.GroupNorm(gcd(ng, n_out), n_out)
        elif norm == 'BN':
            self.bn1 = nn.BatchNorm1d(n_out)
            self.bn2 = nn.BatchNorm1d(n_out)
        else:
            exit('SyncBN has not been added!')

        if stride != 1 or n_out != n_in:
            if norm == 'GN':
                self.downsample = nn.Sequential(
                        nn.Conv1d(n_in, n_out, kernel_size=1, stride=stride, bias=False),
                        nn.GroupNorm(gcd(ng, n_out), n_out))
            elif norm == 'BN':
                self.downsample = nn.Sequential(
                        nn.Conv1d(n_in, n_out, kernel_size=1, stride=stride, bias=False),
                        nn.BatchNorm1d(n_out))
            else:
                exit('SyncBN has not been added!')
        else:
            self.downsample = None

        self.act = act

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x
        if self.act:
            out = self.relu(out)
        return out


class LinearRes(nn.Module):
    def __init__(self, n_in, n_out, norm='GN', ng=32):
        super(LinearRes, self).__init__()
        assert(norm in ['GN', 'BN', 'SyncBN'])

        self.linear1 = nn.Linear(n_in, n_out, bias=False)
        self.linear2 = nn.Linear(n_out, n_out, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if norm == 'GN':
            self.norm1 = nn.GroupNorm(gcd(ng, n_out), n_out)
            self.norm2 = nn.GroupNorm(gcd(ng, n_out), n_out)
        elif norm == 'BN':
            self.norm1 = nn.BatchNorm1d(n_out)
            self.norm2 = nn.BatchNorm1d(n_out)
        else:   
            exit('SyncBN has not been added!')

        if n_in != n_out:
            if norm == 'GN':
                self.transform = nn.Sequential(
                    nn.Linear(n_in, n_out, bias=False),
                    nn.GroupNorm(gcd(ng, n_out), n_out))
            elif norm == 'BN':
                self.transform = nn.Sequential(
                    nn.Linear(n_in, n_out, bias=False),
                    nn.BatchNorm1d(n_out))
            else:
                exit('SyncBN has not been added!')
        else:
            self.transform = None

    def forward(self, x):
        out = self.linear1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.norm2(out)

        if self.transform is not None:
            out += self.transform(x)
        else:
            out += x

        out = self.relu(out)
        return out


class Null(nn.Module):
    def __init__(self):
        super(Null, self).__init__()
        
    def forward(self, x):
        return x


def linear_interp(x, n_max):
    """Given a Tensor of normed positions, returns linear interplotion weights and indices.
    Example: For position 1.2, its neighboring pixels have indices 0 and 1, corresponding
    to coordinates 0.5 and 1.5 (center of the pixel), and linear weights are 0.3 and 0.7.

    Args:
        x: Normalizzed positions, ranges from 0 to 1, float Tensor.
        n_max: Size of the dimension (pixels), multiply x to get absolution positions.
    Returns: Weights and indices of left side and right side.
    """
    x = x * n_max - 0.5

    mask = x < 0
    x[mask] = 0
    mask = x > n_max - 1
    x[mask] = n_max - 1
    n = torch.floor(x)

    rw = x - n
    lw = 1.0 - rw
    li = n.long()
    ri = li + 1
    mask = ri > n_max - 1
    ri[mask] = n_max - 1

    return lw, li, rw, ri


def get_pixel_feat(fm, bboxes, pts_range):
    x, y = bboxes[:, 0], bboxes[:, 1]
    x_min, x_max, y_min, y_max = pts_range[:4]
    x = (x - x_min) / (x_max - x_min)
    y = (y_max - y) / (y_max - y_min)

    _, fm_h, fm_w = fm.size()
    xlw, xli, xhw, xhi = linear_interp(x, fm_w)
    ylw, yli, yhw, yhi = linear_interp(y, fm_h)
    feat = \
        (xlw * ylw).unsqueeze(1) * fm[:, yli, xli].transpose(0, 1) +\
        (xlw * yhw).unsqueeze(1) * fm[:, yhi, xli].transpose(0, 1) +\
        (xhw * ylw).unsqueeze(1) * fm[:, yli, xhi].transpose(0, 1) +\
        (xhw * yhw).unsqueeze(1) * fm[:, yhi, xhi].transpose(0, 1)
    return feat


def get_roi_feat(fm, bboxes, roi_size, pts_range):
    """Given a set of BEV bboxes get their BEV ROI features.

    Args:
        fm: Feature map, float tensor, chw
        bboxes: BEV bboxes, n x 5 float tensor (cx, cy, wid, hgt, theta)
        roi_size: ROI size (number of bins), [int] or int
        pts_range: Range of points, tuple of ints, (x_min, x_max, y_min, y_max, z_min, z_max)
    Returns: Extracted features of size (num_roi, c, roi_size, roi_size).
    """
    if isinstance(roi_size, Number):
        roi_size = [roi_size, roi_size]

    cx, cy, wid, hgt, theta = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3], bboxes[:, 4]
    st = torch.sin(theta)
    ct = torch.cos(theta)
    num_bboxes = len(bboxes)

    rot_mat = bboxes.new().resize_(num_bboxes, 2, 2)
    rot_mat[:, 0, 0] = ct
    rot_mat[:, 0, 1] = -st
    rot_mat[:, 1, 0] = st
    rot_mat[:, 1, 1] = ct

    offset = bboxes.new().resize_(len(bboxes), roi_size[0], roi_size[1], 2)
    x_bin = (torch.arange(roi_size[1]).float().to(bboxes.device) + 0.5) / roi_size[1] - 0.5
    offset[:, :, :, 0] = x_bin.view(1, 1, -1) * wid.view(-1, 1, 1)
    y_bin = (torch.arange(roi_size[0] - 1, -1, -1).float().to(bboxes.device) + 0.5) / roi_size[0] - 0.5
    offset[:, :, :, 1] = y_bin.view(1, -1, 1) * hgt.view(-1, 1, 1)

    rot_mat = rot_mat.view(num_bboxes, 1, 1, 2, 2)
    offset = offset.view(num_bboxes, roi_size[0], roi_size[1], 2, 1)
    offset = torch.matmul(rot_mat, offset).view(num_bboxes, roi_size[0], roi_size[1], 2)

    x = cx.view(-1, 1, 1) + offset[:, :, :, 0]
    y = cy.view(-1, 1, 1) + offset[:, :, :, 1]
    x = x.view(-1)
    y = y.view(-1)

    x_min, x_max, y_min, y_max = pts_range[:4]
    x = (x - x_min) / (x_max - x_min)
    y = (y_max - y) / (y_max - y_min)

    fm_c, fm_h, fm_w = fm.size()
    feat = fm.new().float().resize_(num_bboxes * roi_size[0] * roi_size[1], fm_c)
    mask = (x > 0) * (x < 1) * (y > 0) * (y < 1)
    x = x[mask]
    y = y[mask]

    xlw, xli, xhw, xhi = linear_interp(x, fm_w)
    ylw, yli, yhw, yhi = linear_interp(y, fm_h)
    feat[mask] = \
        (xlw * ylw).unsqueeze(1) * fm[:, yli, xli].transpose(0, 1) +\
        (xlw * yhw).unsqueeze(1) * fm[:, yhi, xli].transpose(0, 1) +\
        (xhw * ylw).unsqueeze(1) * fm[:, yli, xhi].transpose(0, 1) +\
        (xhw * yhw).unsqueeze(1) * fm[:, yhi, xhi].transpose(0, 1)
    feat[torch.logical_not(mask)] = 0
    feat = feat.view(num_bboxes, roi_size[0] * roi_size[1], fm_c)
    feat = feat.transpose(1, 2).contiguous().view(num_bboxes, -1, roi_size[0], roi_size[1])
    return feat
