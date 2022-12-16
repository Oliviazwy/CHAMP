#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import torch.nn as nn
import numpy as np
import torch
from torch.nn.parameter import Parameter
import math
from einops import rearrange
from torch import nn, einsum


class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape  # 32,256,3,24
        qkv = self.to_qkv(x).chunk(3, dim = 1)  # 分块 (32,128,3,24)x3
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)  # rearrange:转换张量维度 q(32,4,32,72) k(32,4,32,72) v(32,4,32,72)
        q = q * self.scale  # (32,4,32,72)

        sim = einsum('b h d i, b h d j -> b h i j', q, k)  # 32,4,72,72
        sim = sim - sim.amax(dim = -1, keepdim = True).detach()  # 32,4,72,72  amax():返回给定维度 dim 中 input 张量的每个切片的最大值。
        attn = sim.softmax(dim = -1)  # 32,4,72,72

        out = einsum('b h i j, b h d j -> b h i d', attn, v)  # 32,4,72,32
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)  # 32,128,3,24
        return self.to_out(out)



class GraphConvolution(nn.Module):
    """
    adapted from : https://github.com/tkipf/gcn/blob/92600c39797c2bfb61a508e52b88fb554df30177/gcn/layers.py#L132
    """

    def __init__(self, in_features, out_features, bias=True, node_n=48):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.att = Parameter(torch.FloatTensor(node_n, node_n))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.att.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(self.att, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GC_Block(nn.Module):
    def __init__(self, in_features, p_dropout, bias=True, node_n=48):
        """
        Define a residual block of GCN
        """
        super(GC_Block, self).__init__()
        self.in_features = in_features
        self.out_features = in_features

        self.gc1 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.bn1 = nn.BatchNorm1d(node_n * in_features)

        self.gc2 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.bn2 = nn.BatchNorm1d(node_n * in_features)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()
        self.attention = Attention(256)

    def forward(self, x):
        y = self.gc1(x)
        b, n, f = y.shape  # 32,72,256
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        y = self.gc2(y)
        b, n, f = y.shape
        y = self.bn2(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)  # 32,72,256
        # add attention
        z = y.transpose(1, 2).reshape(b, 256, 3, -1)  # 32,256,3,24
        y = self.attention(z)  # 32,256,3,24
        y = y.reshape(b, 256, -1).transpose(1, 2)  # 32,72,256

        return y + x

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, input_feature, hidden_feature, p_dropout, num_stage=1, node_n=72,model_prob=None):
        """
        :param input_feature: num of input feature
        :param hidden_feature: num of hidden feature
        :param p_dropout: drop out prob.
        :param num_stage: number of residual blocks
        :param node_n: number of nodes in graph
        """
        super(GCN, self).__init__()
        self.num_stage = num_stage
        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()
        self.idx = model_prob

        # model para
        upper_idx1 = np.array([3, 4, 5, 6, 11, 12, 13, 14, 20, 21])
        lower_idx1 = np.array([0, 1, 2, 7, 8, 9, 10, 15, 16, 17, 18, 19, 22, 23])
        upper_idx2 = np.array([0, 1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 19, 20, 21])
        lower_idx2 = np.array([7, 8, 9, 10, 15, 16, 17, 18, 22, 23])
        upper_idx3 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 21, 22, 23])
        lower_idx3 = np.array([0, 11, 12, 13, 14, 15, 16, 17, 18, 19])
        upper_idx4 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 19, 20, 21, 22, 23])
        lower_idx4 = np.array([0, 11, 12, 13, 14, 15, 16, 17, 18])
        upper_idx5 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 19, 20, 21, 22, 23])
        lower_idx5 = np.array([11, 12, 13, 14, 15, 16, 17, 18])
        self.upper_list = (upper_idx1, upper_idx2, upper_idx3, upper_idx4, upper_idx5)
        self.lower_list = (lower_idx1, lower_idx2, lower_idx3, lower_idx4, lower_idx5)

        # upper part
        node_n1 = self.upper_list[self.idx].size * 3
        self.gc1_1 = GraphConvolution(input_feature, hidden_feature, node_n=node_n1)
        self.bn1 = nn.BatchNorm1d(node_n1 * hidden_feature)

        self.gcbs1 = []
        for i in range(num_stage):
            self.gcbs1.append(GC_Block(hidden_feature, p_dropout=p_dropout, node_n=node_n1))
        self.gcbs1 = nn.ModuleList(self.gcbs1)
        self.gc7_1 = GraphConvolution(hidden_feature, input_feature, node_n=node_n1)

        # lower part
        node_n2 = self.lower_list[self.idx].size * 3
        self.gc1_2 = GraphConvolution(input_feature, hidden_feature, node_n=node_n2)
        self.bn2 = nn.BatchNorm1d(node_n2 * hidden_feature)

        self.gcbs2 = []
        for i in range(num_stage):
            self.gcbs2.append(GC_Block(hidden_feature, p_dropout=p_dropout, node_n=node_n2))
        self.gcbs2 = nn.ModuleList(self.gcbs2)

        self.gc7_2 = GraphConvolution(hidden_feature, input_feature, node_n=node_n2)

        # whole body
        self.gc1 = GraphConvolution(input_feature, hidden_feature, node_n=node_n)
        self.bn = nn.BatchNorm1d(node_n * hidden_feature)

        self.gcbs = []
        for i in range(num_stage):
            self.gcbs.append(GC_Block(hidden_feature, p_dropout=p_dropout, node_n=node_n))

        self.gcbs = nn.ModuleList(self.gcbs)

        self.gc7 = GraphConvolution(hidden_feature, input_feature, node_n=node_n)

    def forward(self, x, data_source=None):
        y_p1 = torch.zeros_like(x)
        y_p2 = torch.zeros_like(x)

        if data_source == 'original' or data_source is None:
            upper_idx = self.upper_list[self.idx]
            upper_idx = np.concatenate((upper_idx * 3, upper_idx * 3 + 1, upper_idx * 3 + 2))
            lower_idx = self.lower_list[self.idx]
            lower_idx = np.concatenate((lower_idx * 3, lower_idx * 3 + 1, lower_idx * 3 + 2))
            upper_idx.sort()
            lower_idx.sort()

            x1 = x[:, upper_idx]
            x2 = x[:, lower_idx]

            # y1: upper part
            y1 = self.gc1_1(x1)
            b1, n1, f1 = y1.shape
            y1 = self.bn1(y1.view(b1, -1)).view(b1, n1, f1)
            y1 = self.act_f(y1)
            y1 = self.do(y1)

            for i in range(self.num_stage):
                y1 = self.gcbs1[i](y1)

            y1 = self.gc7_1(y1)
            y1 = y1 + x1

            # y2: lower part
            y2 = self.gc1_2(x2)  # lower part
            b2, n2, f2 = y2.shape
            y2 = self.bn2(y2.view(b2, -1)).view(b2, n2, f2)
            y2 = self.act_f(y2)
            y2 = self.do(y2)

            for i in range(self.num_stage):
                y2 = self.gcbs2[i](y2)

            y2 = self.gc7_2(y2)
            y2 = y2 + x2

            # 合并
            y_p1 = x
            y_p1[:, upper_idx] = y1
            y_p1[:, lower_idx] = y2

        if data_source == 'augmentation' or data_source is None or data_source == 'original':
            y_p2 = self.gc1(x)
            b, n, f = y_p2.shape
            y_p2 = self.bn(y_p2.view(b, -1)).view(b, n, f)
            y_p2= self.act_f(y_p2)
            y_p2 = self.do(y_p2)

            for i in range(self.num_stage):
                y_p2 = self.gcbs[i](y_p2)

            y_p2 = self.gc7(y_p2)
            y_p2 = y_p2 + x

        return y_p1+y_p2
