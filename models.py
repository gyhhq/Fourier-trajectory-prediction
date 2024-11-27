
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops import rearrange
# import numpy as np
# import argparse
# import os
# from attrdict import AttrDict
# from .sgan1.models1 import TrajectoryGenerator1
# from .sgan1.utils1 import relative_to_abs1
# import matplotlib.pyplot as plt

import sys

def make_mlp(dim_list, activation='relu', batch_norm=True, dropout=0):
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        if batch_norm:
            layers.append(nn.BatchNorm1d(dim_out))
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers).cuda()  # 将模型移动到GPU上


def get_noise(shape, noise_type):
    if noise_type == 'gaussian':
        return torch.randn(*shape).cuda()  # 将张量移动到GPU上
    elif noise_type == 'uniform':
        return torch.rand(*shape).sub_(0.5).mul_(2.0).cuda()  # 将张量移动到GPU上
    raise ValueError('Unrecognized noise type "%s"' % noise_type)

class Encoder(nn.Module):
    def __init__(
        self, embedding_dim=64, h_dim=64, mlp_dim=1024, num_layers=1,
        dropout=0.0,dim=64, decode_dim=64, hidden_size_c=64, mlp_dimension_ds=128,
    ):
        super(Encoder, self).__init__()

        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.encoder = nn.LSTM(embedding_dim, h_dim, num_layers, dropout=dropout).cuda()  # 将LSTM层移动到GPU
        self.encoder2 = nn.LSTM(32, 32, num_layers, dropout=dropout).cuda()  # 将另一个LSTM层移动到GPU
        self.spatial_embedding = nn.Linear(2, embedding_dim).cuda()  # 将Linear层移动到GPU

        self.to_qkv = nn.Linear(dim, decode_dim * 3, bias=False).cuda()  # 将Linear层移动到GPU
        self.weight_q = nn.Linear(dim, decode_dim, bias=False).cuda()  # 将Linear层移动到GPU
        self.weight_k = nn.Linear(dim, decode_dim, bias=False).cuda()  # 将Linear层移动到GPU
        self.weight_v = nn.Linear(dim, decode_dim, bias=False).cuda()  # 将Linear层移动到GPU
        self.weight_r = nn.Linear(decode_dim, decode_dim, bias=False).cuda()  # 将Linear层移动到GPU
        self.weight_alpha = nn.Parameter(torch.randn(decode_dim).cuda())  # 将参数移动到GPU
        self.weight_beta = nn.Parameter(torch.randn(decode_dim).cuda())  # 将参数移动到GPU
        self.scale_factor = decode_dim ** -0.5

        self.num_layers = num_layers
        self.hidden_size_c = hidden_size_c
        self.mlp_dimension_ds = mlp_dimension_ds
        self.linear = nn.Linear(in_features=64, out_features=32).cuda()  # 将Linear层移动到GPU
        self.linear1 = nn.Linear(in_features=128, out_features=64).cuda()  # 将Linear层移动到GPU

        # self.attention=nn.MultiheadAttention(embed_dim=64,num_heads=4)

    def init_hidden(self, batch):
        return (
               torch.zeros(self.num_layers, batch, self.h_dim).cuda(),
               torch.zeros(self.num_layers, batch, self.h_dim).cuda()
        )

    def forward(self, obs_traj):
        batch = obs_traj.size(1)
        obs_traj_embedding = self.spatial_embedding(obs_traj.reshape(-1, 2))
        obs_traj_embedding = obs_traj_embedding.view(-1, batch, self.embedding_dim)
        state_tuple = self.init_hidden(batch)

        x = obs_traj_embedding
        seq_length_s = x.size(1)

        x = x.cuda()
        state_tuple = tuple([tensor.cuda() for tensor in state_tuple])

        # 傅里叶注意力
        model = mlp_layer(seq_length_s, self.hidden_size_c, self.embedding_dim, self.mlp_dimension_ds)
        query = self.weight_q(x)
        query = model(query)

        key = self.weight_k(x)

        value = self.weight_v(x)
        # value = model(value)

        b, n, d = query.shape
        mask = torch.ones(1, 64).bool()
        mask=mask.cuda()
        mask_value = -torch.finfo(x.dtype).max
        mask = rearrange(mask, 'b n -> b () n')


        # Caculate the global query
        alpha_weight = (torch.mul(query, self.weight_alpha) * self.scale_factor).masked_fill(~mask, mask_value)
        alpha_weight = torch.softmax(alpha_weight, dim=-1)
        global_query = query * alpha_weight
        global_query = torch.einsum('b n d -> b d', global_query)

        # Model the interaction between global query vector and the key vector
        repeat_global_query = einops.repeat(global_query, 'b d -> b copy d', copy=n)
        p = repeat_global_query * key
        p = model(p)
        beta_weight = (torch.mul(p, self.weight_beta) * self.scale_factor).masked_fill(~mask, mask_value)
        beta_weight = torch.softmax(beta_weight, dim=-1)
        global_key = p * beta_weight
        global_key = torch.einsum('b n d -> b d', global_key)

        # key-value
        key_value_interaction = torch.einsum('b j, b n j -> b n j', global_key, value)
        key_value_interaction_out = self.weight_r(key_value_interaction)
        result = key_value_interaction_out + query

        output, state = self.encoder(result, state_tuple)
        obs_traj_embedding=self.linear(obs_traj_embedding)
        output=output+obs_traj_embedding
        final_h = output[-1]

        # batch = obs_traj.size(1)
        # obs_traj_embedding = self.spatial_embedding(obs_traj.reshape(-1, 2))
        # obs_traj_embedding = obs_traj_embedding.view(
        #     -1, batch, self.embedding_dim)
        # state_tuple = self.init_hidden(batch)
        # output, state = self.encoder(obs_traj_embedding, state_tuple)
        # final_h = state[0]


        return final_h


# class Encoder1(nn.Module):
#     def __init__(
#         self, embedding_dim=64, h_dim=64, mlp_dim=1024, num_layers=1,
#         dropout=0.0,dim=64, decode_dim=64, hidden_size_c=64, mlp_dimension_ds=128,
#     ):
#         super(Encoder1, self).__init__()
#
#         self.mlp_dim = mlp_dim
#         self.h_dim = h_dim
#         self.embedding_dim = embedding_dim
#         self.num_layers = num_layers
#         self.encoder = nn.LSTM(embedding_dim, 48, num_layers, dropout=dropout).cuda()  # 将LSTM层移动到GPU
#         self.encoder2 = nn.LSTM(32, 32, num_layers, dropout=dropout).cuda()  # 将另一个LSTM层移动到GPU
#         self.spatial_embedding = nn.Linear(2, embedding_dim).cuda()  # 将Linear层移动到GPU
#
#         self.to_qkv = nn.Linear(dim, decode_dim * 3, bias=False).cuda()  # 将Linear层移动到GPU
#         self.weight_q = nn.Linear(dim, decode_dim, bias=False).cuda()  # 将Linear层移动到GPU
#         self.weight_k = nn.Linear(dim, decode_dim, bias=False).cuda()  # 将Linear层移动到GPU
#         self.weight_v = nn.Linear(dim, decode_dim, bias=False).cuda()  # 将Linear层移动到GPU
#         self.weight_r = nn.Linear(decode_dim, decode_dim, bias=False).cuda()  # 将Linear层移动到GPU
#         self.weight_alpha = nn.Parameter(torch.randn(decode_dim).cuda())  # 将参数移动到GPU
#         self.weight_beta = nn.Parameter(torch.randn(decode_dim).cuda())  # 将参数移动到GPU
#         self.scale_factor = decode_dim ** -0.5
#
#         self.num_layers = num_layers
#         self.hidden_size_c = hidden_size_c
#         self.mlp_dimension_ds = mlp_dimension_ds
#         self.linear = nn.Linear(in_features=64, out_features=48).cuda()  # 将Linear层移动到GPU
#         self.linear1 = nn.Linear(in_features=128, out_features=64).cuda()  # 将Linear层移动到GPU
#
#         # self.attention=nn.MultiheadAttention(embed_dim=64,num_heads=4)
#
#     def init_hidden(self, batch):
#         return (
#                torch.zeros(self.num_layers, batch, self.h_dim).cuda(),
#                torch.zeros(self.num_layers, batch, self.h_dim).cuda()
#         )
#
#     def forward(self, obs_traj):
#         batch = obs_traj.size(1)
#         obs_traj_embedding = self.spatial_embedding(obs_traj.reshape(-1, 2))
#         obs_traj_embedding = obs_traj_embedding.view(-1, batch, self.embedding_dim)
#         state_tuple = self.init_hidden(batch)
#
#         x = obs_traj_embedding
#         seq_length_s = x.size(1)
#
#         x = x.cuda()
#         state_tuple = tuple([tensor.cuda() for tensor in state_tuple])
#
#         # 傅里叶注意力
#         model = mlp_layer(seq_length_s, self.hidden_size_c, self.embedding_dim, self.mlp_dimension_ds)
#         query = self.weight_q(x)
#         query = model(query)
#
#         key = self.weight_k(x)
#
#         value = self.weight_v(x)
#         # value = model(value)
#
#         b, n, d = query.shape
#         mask = torch.ones(1, 64).bool()
#         mask=mask.cuda()
#         mask_value = -torch.finfo(x.dtype).max
#         mask = rearrange(mask, 'b n -> b () n')
#
#
#         # Caculate the global query
#         alpha_weight = (torch.mul(query, self.weight_alpha) * self.scale_factor).masked_fill(~mask, mask_value)
#         alpha_weight = torch.softmax(alpha_weight, dim=-1)
#         global_query = query * alpha_weight
#         global_query = torch.einsum('b n d -> b d', global_query)
#
#         # Model the interaction between global query vector and the key vector
#         repeat_global_query = einops.repeat(global_query, 'b d -> b copy d', copy=n)
#         p = repeat_global_query * key
#         p = model(p)
#         beta_weight = (torch.mul(p, self.weight_beta) * self.scale_factor).masked_fill(~mask, mask_value)
#         beta_weight = torch.softmax(beta_weight, dim=-1)
#         global_key = p * beta_weight
#         global_key = torch.einsum('b n d -> b d', global_key)
#
#         # key-value
#         key_value_interaction = torch.einsum('b j, b n j -> b n j', global_key, value)
#         key_value_interaction_out = self.weight_r(key_value_interaction)
#         result = key_value_interaction_out + query
#
#         output, state = self.encoder(result, state_tuple)
#         obs_traj_embedding=self.linear(obs_traj_embedding)
#         output=output+obs_traj_embedding
#         final_h = output[-1]
#
#         # batch = obs_traj.size(1)
#         # obs_traj_embedding = self.spatial_embedding(obs_traj.reshape(-1, 2))
#         # obs_traj_embedding = obs_traj_embedding.view(
#         #     -1, batch, self.embedding_dim)
#         # state_tuple = self.init_hidden(batch)
#         # output, state = self.encoder(obs_traj_embedding, state_tuple)
#         # final_h = state[0]
#
#
#         return final_h

class Encoder1(nn.Module):
    """Encoder is part of both TrajectoryGenerator and
    TrajectoryDiscriminator"""
    def __init__(
        self, embedding_dim=64, h_dim=64, mlp_dim=1024, num_layers=1,
        dropout=0.0
    ):
        super(Encoder1, self).__init__()

        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.encoder = nn.LSTM(embedding_dim, h_dim, num_layers, dropout=dropout).cuda()
        self.spatial_embedding = nn.Linear(2, embedding_dim).cuda()

    def init_hidden(self, batch):
        return (
               torch.zeros(self.num_layers, batch, self.h_dim).cuda(),
               torch.zeros(self.num_layers, batch, self.h_dim).cuda()
        )

    def forward(self, obs_traj):
        batch = obs_traj.size(1)
        obs_traj_embedding = self.spatial_embedding(obs_traj.reshape(-1, 2))

        obs_traj_embedding = obs_traj_embedding.view(
            -1, batch, self.embedding_dim)
        state_tuple = self.init_hidden(batch)

        output, state = self.encoder(obs_traj_embedding, state_tuple)
        final_h = state[0]
        return final_h

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim).cuda(),
            nn.GELU().cuda(),
            nn.Dropout(dropout).cuda(),
            nn.Linear(hidden_dim, dim).cuda(),
            nn.Dropout(dropout).cuda()
        )

    def forward(self, x):
        return self.net(x)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim).cuda()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FNetBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x_real = x.unsqueeze(-1)
        x_imag = torch.zeros_like(x_real).cuda()
        x_complex = torch.complex(x_real, x_imag)
        result_complex = torch.fft.fft(torch.fft.fft(x_complex, dim=-1), dim=-2)
        return result_complex


class FNet(nn.Module):
    def __init__(self, dim, depth, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, FNetBlock()).cuda(),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)).cuda()
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x1 = x.to(torch.float32)
            x_a = attn(x1).squeeze(-1)
            x = x_a + x
            x = x.to(torch.complex64)
            x2 = x.to(torch.float32)
            x = ff(x2) + x
        return x

class FeedForward1(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim).cuda(),
            nn.GELU().cuda(),
            nn.Dropout(dropout).cuda(),
            nn.Linear(hidden_dim, dim).cuda(),
            nn.Dropout(dropout).cuda()
        )

    def forward(self, x):
        return self.net(x)


class PreNorm1(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim).cuda()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FNetBlock1(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x_real = x.unsqueeze(-1)
        x_imag = torch.zeros_like(x_real).cuda()
        x_complex = torch.complex(x_real, x_imag)
        result_complex = torch.fft.fft(torch.fft.fft(x_complex, dim=-1), dim=-2)
        return result_complex

class FNet1(nn.Module):
    def __init__(self, dim, depth, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm1(dim, FNetBlock1()).cuda(),
                PreNorm1(dim, FeedForward1(dim, mlp_dim, dropout=dropout)).cuda()
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x1 = x.to(torch.float32)
            x_a = attn(x1).squeeze(-1)
            x = x_a + x
            x = x.to(torch.complex64)
            x2 = x.to(torch.float32)
            x = ff(x2) + x
        return x

class FeedForward2(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim).cuda(),
            nn.GELU().cuda(),
            nn.Dropout(dropout).cuda(),
            nn.Linear(hidden_dim, dim).cuda(),
            nn.Dropout(dropout).cuda()
        )

    def forward(self, x):
        return self.net(x)

class PreNorm2(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim).cuda()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FNetBlock2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.fft.fft(torch.fft.fft(x, dim=-1), dim=-2).real
        return x

class InverseFNetBlock2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.fft.ifft(torch.fft.ifft(x, dim=-1), dim=-2).real
        return x

class FNet2(nn.Module):
    def __init__(self, dim, depth, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm2(dim, FNetBlock2()).cuda(),
                PreNorm2(dim, FeedForward2(dim, mlp_dim, dropout=dropout)).cuda()
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x1 = x.to(torch.float32)
            x_a = attn(x1).squeeze(-1)
            x = x_a + x
            x = x.to(torch.complex64)
            x2 = x.to(torch.float32)
            x = ff(x2) + x
        return x

class mlp_block(nn.Module):
    def __init__(self, in_channels, mlp_dim, drop_ratio=0.):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_channels, mlp_dim).cuda(),
            nn.GELU().cuda(),
            nn.Dropout(drop_ratio).cuda(),
            nn.Linear(mlp_dim, in_channels).cuda(),
            nn.Dropout(drop_ratio).cuda()
        )

    def forward(self, x):
        x = self.block(x)
        return x

class mlp_layer(nn.Module):
    def __init__(self, seq_length_s, hidden_size_c, mlp_dimension_dc, mlp_dimension_ds, embedding_dim=64, h_dim=64, num_layers=1, dropout=0.0):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_size_c).cuda()
        # 添加多头注意力层
        self.num_layers = num_layers
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.spatial_embedding = nn.Linear(2, embedding_dim).cuda()
        self.encoder = nn.LSTM(embedding_dim, h_dim, num_layers, dropout=dropout).cuda()
        self.attention = nn.MultiheadAttention(embed_dim=64, num_heads=4).cuda()

    def init_hidden(self, batch):
        return (
            torch.zeros(self.num_layers, batch, self.h_dim).cuda(),
            torch.zeros(self.num_layers, batch, self.h_dim).cuda()
        )

    def forward(self, x):
        x = x.cuda()
        x1 = self.ln(x)   # 归一化

        batch = x1.size(1)
        state_tuple = self.init_hidden(batch)
        output0, state0 = self.encoder(x1, state_tuple)   # gt
        output0A, _ = self.attention(output0, output0, output0)

        x2 = output0[-1]
        x2 = torch.unsqueeze(x2, dim=0)
        batch0 = x2.size(1)
        state_tuple1 = self.init_hidden(batch0)
        output1, state1 = self.encoder(x2, state_tuple1)   # gf

        x3 = output0A[-1]
        x3 = torch.unsqueeze(x3, dim=0)
        batch = x3.size(1)
        state_tuple = self.init_hidden(batch)
        output0AA, state0AA = self.encoder(x3, state_tuple)   # gf ATTENTION


        tensor = output0
        tensor_3 = output1
        sub_tensors = []

        tensorA = output0A
        tensor_3A = output0AA

        for i in range(tensor.size(0)):
            tensor_2 = tensor_3
            tensor_2 = tensor_2.transpose(1, 2)  # 转置矩阵
            tensor_2 = tensor_2.squeeze(0)
            tensor_2=tensor_2.unsqueeze(0)
            dim = tensor_2.size(2)
            model1 = FNet1(dim, depth=4, mlp_dim=512).cuda()  # input_tensor = torch.randn(8, 32, 256)
            tensor_2=model1(tensor_2)  # 傅里叶变换

            sub_tensor = tensor[i:i + 1]  # 取出hang当前位置的子张量
            sub_tensor = sub_tensor.squeeze(0)
            sub_tensor= sub_tensor.unsqueeze(0)
            dim = sub_tensor.size(2)
            model2 = FNet1(dim, depth=4, mlp_dim=512).cuda()  # input_tensor = torch.randn(8, 32, 256)
            sub_tensor = model2(sub_tensor)
            result = torch.matmul(sub_tensor, tensor_2)
            tensor_2=tensor_2.squeeze(0)
            result=result.squeeze(0)
            result3= torch.matmul(tensor_2,result)
            # result3 = result3.unsqueeze(0)


            tensor_2A = tensor_3A
            tensor_2A = tensor_2A.transpose(1, 2)  # 转置矩阵
            tensor_2A = tensor_2A.squeeze(0)
            tensor_2A=tensor_2A.unsqueeze(0)
            dimA = tensor_2A.size(2)
            model1A = FNet1(dimA, depth=4, mlp_dim=512).cuda()  # input_tensor = torch.randn(8, 32, 256)
            tensor_2A=model1A(tensor_2A)  # 傅里叶变换

            sub_tensorA = tensorA[i:i + 1]  # 取出当前位置的子张量
            sub_tensorA = sub_tensorA.squeeze(0)
            sub_tensorA= sub_tensorA.unsqueeze(0)
            dimA = sub_tensorA.size(2)
            model2A = FNet1(dimA, depth=4, mlp_dim=512).cuda()  # input_tensor = torch.randn(8, 32, 256)
            sub_tensorA = model2A(sub_tensorA)

            resultA = torch.matmul(sub_tensorA, tensor_2A)
            tensor_2A=tensor_2A.squeeze(0)
            resultA=resultA.squeeze(0)

            result3A = torch.matmul(tensor_2A,resultA)
            result3A = result3A.unsqueeze(0)



            # result3 = torch.mul(result3A, result3)    #时空相乘
            result3=result3.unsqueeze(0)      #仅时间
            # result3 = result3A       #仅空间




            dim = result3.size(2)
            model2 = FNet2(dim, depth=4, mlp_dim=64, dropout=0.1).cuda()  # 反傅里叶变换
            result3 = model2(result3)
            result3 = torch.fft.ifft(torch.fft.ifft(result3, dim=-1), dim=-2).real

            normalized_data = F.normalize(result3, p=2, dim=1)  # 调用 normalize 函数对数据进行归一化处理
            result3 = F.softmax(normalized_data, dim=-1)
            result3=result3.squeeze(0)

            sub_tensors.append(result3)  # 将子张量添加到列表中
        results_tensor = torch.stack(sub_tensors, dim=0)
        results_tensor = results_tensor.transpose(1, 2)  # 转置矩阵

        # # 将矩阵保存到文件中
        # with open('matrix_output1111.txt', 'w') as f:
        #     for row in results_tensor[-1]:
        #         f.write(' '.join(map(str, row)) + '\n')

        return results_tensor


# class mlp_layer(nn.Module):
#     def __init__(self, seq_length_s, hidden_size_c, mlp_dimension_dc, mlp_dimension_ds, embedding_dim=64, h_dim=64, num_layers=1, dropout=0.0):
#         super().__init__()
#         self.ln = nn.LayerNorm(hidden_size_c).cuda()
#
#         self.num_layers = num_layers
#         self.h_dim = h_dim
#         self.embedding_dim = embedding_dim
#         self.spatial_embedding = nn.Linear(2, embedding_dim).cuda()
#         self.encoder = nn.LSTM(embedding_dim, h_dim, num_layers, dropout=dropout).cuda()
#
#     def init_hidden(self, batch):
#         return (
#             torch.zeros(self.num_layers, batch, self.h_dim).cuda(),
#             torch.zeros(self.num_layers, batch, self.h_dim).cuda()
#         )
#     def forward(self, x):
#         x = x.cuda()
#         x1 = self.ln(x)   # 归一化
#         dim = x1.size(2)
#         model1 = FNet1(dim, depth=4, mlp_dim=512).cuda()  # input_tensor = torch.randn(8, 32, 256)
#         x1 = model1(x1)  # 傅里叶变换
#
#         batch = x1.size(1)
#         state_tuple = self.init_hidden(batch)
#         output0, state0 = self.encoder(x1, state_tuple)   # gt
#         print(output0.shape)
#
#         x2 = output0[-1]
#         x2 = torch.unsqueeze(x2, dim=0)
#         batch0 = x2.size(1)
#         state_tuple1 = self.init_hidden(batch0)
#         output1, state1 = self.encoder(x2, state_tuple1)   # gf
#         print(output1.shape)
#
#         tensor = output0
#         tensor_3 = output1
#         sub_tensors = []
#
#         for i in range(tensor.size(0)):
#             tensor_2 = tensor_3
#             sub_tensor = tensor[i:i + 1]  # 取出当前位置的子张量
#             tensor_2 = tensor_2.transpose(1, 2)  # 转置矩阵
#             sub_tensor = sub_tensor.squeeze(0)
#             tensor_2 = tensor_2.squeeze(0)
#
#             result = torch.matmul(sub_tensor, tensor_2)
#             result1 = torch.matmul(tensor_2, sub_tensor)
#
#             line1 = nn.Linear(result.shape[0], 1, bias=True).cuda()
#             result = line1(result)
#             line2 = nn.Linear(result1.shape[0], 1, bias=True).cuda()
#             result1 = line2(result1)
#
#             result1 = result1.transpose(1, 0)  # 转置矩阵
#             result3 = torch.matmul(result, result1)
#
#             sub_tensors.append(result3)  # 将子张量添加到列表中
#         results_tensor = torch.stack(sub_tensors, dim=0)
#
#         x7 = self.model2(results_tensor)
#         x7 = torch.fft.ifft(torch.fft.ifft(x7, dim=-1), dim=-2).real
#         x7 = x7.reshape(-1, x7.size(-1))
#         normalized_data = F.normalize(x7, p=2, dim=1)  # 调用normalize函数对数据进行归一化处理
#         x7 = F.softmax(normalized_data, dim=-1)
#         x7_softmax = x7.view(x.size(0), x.size(1), -1)
#
#         return x7_softmax

# class Encoder1(nn.Module):
#     """Encoder is part of both TrajectoryGenerator and
#     TrajectoryDiscriminator"""
#     def __init__(
#         self, embedding_dim=64, h_dim=64, mlp_dim=1024, num_layers=1,
#         dropout=0.0
#     ):
#         super(Encoder1, self).__init__()
#
#         self.mlp_dim = mlp_dim
#         self.h_dim = h_dim
#         self.embedding_dim = embedding_dim
#         self.num_layers = num_layers
#
#         self.encoder = nn.LSTM(
#             embedding_dim, h_dim, num_layers, dropout=dropout
#         ).cuda()  # Move encoder to GPU
#         self.spatial_embedding = nn.Linear(2, embedding_dim).cuda()  # Move spatial_embedding to GPU
#
#     def init_hidden(self, batch):
#         return (
#                torch.zeros(self.num_layers, batch, self.h_dim).cuda(),  # Move tensor to GPU
#                torch.zeros(self.num_layers, batch, self.h_dim).cuda()  # Move tensor to GPU
#         )
#
#     def forward(self, obs_traj):
#         batch = obs_traj.size(1)
#         obs_traj_embedding = self.spatial_embedding(obs_traj.reshape(-1, 2))
#         obs_traj_embedding = obs_traj_embedding.view(
#             -1, batch, self.embedding_dim)
#         state_tuple = self.init_hidden(batch)
#         output, state = self.encoder(obs_traj_embedding, state_tuple)
#         final_h = state[0]
#         return final_h

class Decoder(nn.Module):
    """Decoder is part of TrajectoryGenerator"""
    def __init__(
        self, seq_len, embedding_dim=64, h_dim=128, mlp_dim=1024, num_layers=1,
        pool_every_timestep=True, dropout=0.0, bottleneck_dim=1024,
        activation='relu', batch_norm=True, pooling_type='pool_net',
        neighborhood_size=2.0, grid_size=8
    ):
        super(Decoder, self).__init__()

        self.seq_len = seq_len
        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.pool_every_timestep = pool_every_timestep

        self.decoder = nn.LSTM(
            embedding_dim, h_dim, num_layers, dropout=dropout
        ).cuda()  # Move decoder to GPU

        if pool_every_timestep:
            if pooling_type == 'pool_net':
                self.pool_net = PoolHiddenNet(
                    embedding_dim=self.embedding_dim,
                    h_dim=self.h_dim,
                    mlp_dim=mlp_dim,
                    bottleneck_dim=bottleneck_dim,
                    activation=activation,
                    batch_norm=batch_norm,
                    dropout=dropout
                ).cuda()  # Move pool_net to GPU


            mlp_dims = [h_dim + bottleneck_dim, mlp_dim, h_dim]
            self.mlp = make_mlp(
                mlp_dims,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout
            ).cuda()  # Move mlp to GPU

        self.spatial_embedding = nn.Linear(2, embedding_dim).cuda()  # Move spatial_embedding to GPU

        self.hidden2pos = nn.Linear(h_dim, 2).cuda()   # Move hidden2pos to GPU


    def forward(self, last_pos, last_pos_rel, state_tuple, seq_start_end):
        batch = last_pos.size(0)
        pred_traj_fake_rel = []
        decoder_input = self.spatial_embedding(last_pos_rel) # 将行人最后的位置点enbedding化，做为decoder输入
        decoder_input = decoder_input.view(1, batch, self.embedding_dim)  # 重塑操作将decoder_input从原来的形状(5, 64)转换为新的形状(1, 5, 64)

        for _ in range(self.seq_len):
            state_tuple_c=state_tuple[1].cuda()
            state_tuple=(state_tuple[0],state_tuple_c)
            output, state_tuple = self.decoder(decoder_input, state_tuple)
            rel_pos = self.hidden2pos(output.view(-1, self.h_dim))
            curr_pos = rel_pos + last_pos   #last_pos表示上一个时刻的位置信息，而rel_pos表示相对位置信息，两者相加可以得到当前位置的估计。

            if self.pool_every_timestep:  #这段代码的主要功能是对解码器的隐藏状态进行处理和更新
                decoder_h = state_tuple[0] #隐藏状态
                pool_h = self.pool_net(decoder_h, seq_start_end, curr_pos)  #池化后的结果
                decoder_h = torch.cat(
                    [decoder_h.view(-1, self.h_dim), pool_h], dim=1)  #拼接
                decoder_h = self.mlp(decoder_h)  #
                decoder_h = torch.unsqueeze(decoder_h, 0)
                state_tuple = (decoder_h, state_tuple[1])   #更新解码器的隐藏状态

            embedding_input = rel_pos

            decoder_input = self.spatial_embedding(embedding_input)
            decoder_input = decoder_input.view(1, batch, self.embedding_dim)
            pred_traj_fake_rel.append(rel_pos.view(batch, -1))
            last_pos = curr_pos

        pred_traj_fake_rel = torch.stack(pred_traj_fake_rel, dim=0)
        return pred_traj_fake_rel, state_tuple[0]

class PoolHiddenNet(nn.Module):
    def __init__(
        self, embedding_dim=64, h_dim=32, mlp_dim=1024, bottleneck_dim=1024,
        activation='relu', batch_norm=True, dropout=0.0
    ):
        super(PoolHiddenNet, self).__init__()
        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.bottleneck_dim = bottleneck_dim
        self.embedding_dim = embedding_dim

        mlp_pre_dim = embedding_dim + h_dim  # Position information's embedding dimension and hidden state's dimension
        mlp_pre_pool_dims = [mlp_pre_dim, 512,
                             bottleneck_dim]  # mlp_pre_dim is the output dimension of the previous layer, 512 is the dimension of the hidden layer, bottleneck_dim is the output dimension of the last layer
        self.spatial_embedding = nn.Linear(2,
                                           embedding_dim).cuda()  # Linear transformation layer to map input spatial position data of dimension 2 to feature representation of dimension embedding_dim
        self.mlp_pre_pool = make_mlp(
            mlp_pre_pool_dims,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout).cuda()  # Multilayer perceptron model

    def repeat(self, tensor, num_reps):
        col_len = tensor.size(1)
        tensor = tensor.unsqueeze(dim=1).repeat(1, num_reps, 1)
        tensor = tensor.view(-1, col_len)
        return tensor.cuda()

    def forward(self, h_states, seq_start_end, end_pos):
        pool_h = []
        for _, (start, end) in enumerate(seq_start_end):
            start = start.item()
            end = end.item()
            num_ped = end - start
            curr_hidden = h_states.reshape(-1, self.h_dim)[start:end].cuda()
            curr_end_pos = end_pos[start:end].cuda()

            curr_hidden_1 = curr_hidden.repeat(num_ped, 1)
            curr_end_pos_1 = curr_end_pos.repeat(num_ped, 1)
            curr_end_pos_2 = self.repeat(curr_end_pos, num_ped)
            curr_rel_pos = curr_end_pos_1 - curr_end_pos_2

            # Normalize the relative position difference
            curr_rel_pos = F.normalize(curr_rel_pos, p=2, dim=-1)

            # Original code remains
            curr_rel_embedding = self.spatial_embedding(curr_rel_pos)
            mlp_h_input = torch.cat([curr_rel_embedding, curr_hidden_1], dim=1)

            # Original code remains
            curr_pool_h = self.mlp_pre_pool(mlp_h_input)
            curr_pool_h = curr_pool_h.view(num_ped, num_ped, -1).max(1)[0]
            pool_h.append(curr_pool_h)

        pool_h = torch.cat(pool_h, dim=0)
        return pool_h



# #运动趋势预测
# parser = argparse.ArgumentParser()
# parser.add_argument('--model_path',default='/home/sayu/pycharmproject/human_machine cooperate/scripts/sgan/sgan1/prediction data for revealing trajectory',type=str)
# parser.add_argument('--num_samples', default=1, type=int)
# def get_generator(checkpoint):
#     args1 = AttrDict(checkpoint['args'])
#     generator = TrajectoryGenerator1(
#         obs_len=args1.obs_len,
#         pred_len=args1.pred_len,
#         embedding_dim=args1.embedding_dim,
#         encoder_h_dim=args1.encoder_h_dim_g,
#         decoder_h_dim=args1.decoder_h_dim_g,
#         mlp_dim=args1.mlp_dim,
#         num_layers=args1.num_layers,
#         noise_dim=args1.noise_dim,
#         noise_type=args1.noise_type,
#         noise_mix_type=args1.noise_mix_type,
#         pooling_type=args1.pooling_type,
#         pool_every_timestep=args1.pool_every_timestep,
#         dropout=args1.dropout,
#         bottleneck_dim=args1.bottleneck_dim,
#         neighborhood_size=args1.neighborhood_size,
#         grid_size=args1.grid_size,
#         batch_norm=args1.batch_norm)
#     generator.load_state_dict(checkpoint['g_state'])   #通过加载训练后的模型的状态字典，将预训练的权重加载到生成器模型中
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     generator.to(device)  # 将生成器模型移动到设备上进行计算
#     # generator.cuda()   #将生成器模型移动到GPU上进行推理。这表明模型已经训练过，并且可以在GPU上进行计算。
#     generator.train()
#     return generator
#
# def evaluate(generator,obs_traj, obs_traj_rel, seq_start_end):
#     with torch.no_grad():
#         pred_traj_fake_rel = generator(
#             obs_traj, obs_traj_rel, seq_start_end
#         )
#         pred_traj_fake = relative_to_abs1(
#             pred_traj_fake_rel, obs_traj[-1]
#         )
#
#         return pred_traj_fake, pred_traj_fake_rel
#
#
# def prediction(args,obs_traj, obs_traj_rel, seq_start_end):
#     if os.path.isdir(args.model_path):
#         filenames = os.listdir(args.model_path)
#         filenames.sort()
#         paths = [
#             os.path.join(args.model_path, file_) for file_ in filenames
#         ]
#     else:
#         paths = [args.model_path]
#     for path in paths:
#         checkpoint = torch.load(path)
#         if 'g_state' not in checkpoint:
#             continue  # 如果没有 g_state 键，则继续下一个循环
#         generator = get_generator(checkpoint)
#
#         pred_traj_fake1, pred_traj_fake_rel1 = evaluate( generator,obs_traj, obs_traj_rel, seq_start_end )
#         break
#     return pred_traj_fake1, pred_traj_fake_rel1




class TrajectoryGenerator(nn.Module):
    def __init__(
        self, obs_len, pred_len, embedding_dim=64, encoder_h_dim=64,
        decoder_h_dim=128, mlp_dim=1024, num_layers=1, noise_dim=(0, ),
        noise_type='gaussian', noise_mix_type='ped', pooling_type=None,
        pool_every_timestep=True, dropout=0.0006, bottleneck_dim=1024,
        activation='relu', batch_norm=True, neighborhood_size=2.0, grid_size=16,
        transformer_hidden_dim=32,num_heads=8
    ):
        super(TrajectoryGenerator, self).__init__()

        if pooling_type and pooling_type.lower() == 'none':
            pooling_type = None

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.mlp_dim = mlp_dim
        self.encoder_h_dim = encoder_h_dim
        self.decoder_h_dim = decoder_h_dim
        self.embedding_dim = embedding_dim
        self.noise_dim = noise_dim
        self.num_layers = num_layers
        self.noise_type = noise_type
        self.noise_mix_type = noise_mix_type
        self.pooling_type = pooling_type
        self.noise_first_dim = 0
        self.pool_every_timestep = pool_every_timestep
        self.bottleneck_dim = 1024
        self.transformer_hidden_dim=transformer_hidden_dim
        self.num_heads=num_heads

        self.encoder = Encoder(
            embedding_dim=embedding_dim,
            h_dim=encoder_h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        self.decoder = Decoder(
            pred_len,
            embedding_dim=embedding_dim,
            h_dim=decoder_h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            pool_every_timestep=pool_every_timestep,
            dropout=dropout,
            bottleneck_dim=bottleneck_dim,
            activation=activation,
            batch_norm=batch_norm,
            pooling_type=pooling_type,
            grid_size=grid_size,
            neighborhood_size=neighborhood_size
        )

        if pooling_type == 'pool_net':
            self.pool_net = PoolHiddenNet(
                embedding_dim=self.embedding_dim,
                h_dim=encoder_h_dim,
                mlp_dim=mlp_dim,
                bottleneck_dim=bottleneck_dim,
                activation=activation,
                batch_norm=batch_norm
            )

        if self.noise_dim == None:
            t=1
        elif self.noise_dim[0] == 0:
            self.noise_dim = None
        else:
            self.noise_first_dim = noise_dim[0]

        # Decoder Hidden
        if pooling_type:
            input_dim = encoder_h_dim + bottleneck_dim
        else:
            input_dim = encoder_h_dim

        if self.mlp_decoder_needed():
            mlp_decoder_context_dims = [
                input_dim, mlp_dim, decoder_h_dim - self.noise_first_dim
            ]

            self.mlp_decoder_context = make_mlp(
                mlp_decoder_context_dims,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout
            )

    def add_noise(self, _input, seq_start_end, user_noise=None):   # 用于给输入数据添加噪声
        if not self.noise_dim:  # 检查 self.noise_dim 是否存在。如果不存在（为 None），则直接返回输入 _input，不进行噪声添加操作。
            return _input

        if self.noise_mix_type == 'global':
            noise_shape = (seq_start_end.size(0), ) + self.noise_dim
        else:
            noise_shape = (_input.size(0), ) + self.noise_dim

        if user_noise is not None:  # 如果提供了 user_noise，则将 z_decoder 设置为 user_noise，否则使用 get_noise 函数生成噪声张量 z_decoder，其形状与上一步确定的噪声形状相同。
            z_decoder = user_noise
        else:
            z_decoder = get_noise(noise_shape, self.noise_type)

        if self.noise_mix_type == 'global':  #global全局
            _list = []
            for idx, (start, end) in enumerate(seq_start_end):  # 循环遍历 seq_start_end 列表中的元组，其中每个元组包含序列的起始和结束位置
                start = start.item()
                end = end.item()
                _vec = z_decoder[idx].view(1, -1)  #从噪声张量 z_decoder 中获取索引为 idx 的张量 _vec，并使用 .view(1, -1) 将其变形为形状为 (1, 噪声维度) 的张量。 结果是1行N列
                _to_cat = _vec.repeat(end - start, 1)  #repeat后变成 end - start行N列
                _list.append(torch.cat([_input[start:end], _to_cat], dim=1)) #从输入数据中选取的特征部分 _input[start:end] 与对应的噪声部分 _to_cat 进行拼接，得到一个包含特征和噪声的张量。
            decoder_h = torch.cat(_list, dim=0)   #得到一个合理的噪声
            return decoder_h

        decoder_h = torch.cat([_input, z_decoder], dim=1) #将输入和噪声进行拼接,dim=1表示在第1维度上进行拼接，即在列方向上进行拼接。

        return decoder_h

    def mlp_decoder_needed(self): #它判断是否需要使用多层感知机（MLP）作为解码器
        if (
            self.noise_dim or self.pooling_type or
            self.encoder_h_dim != self.decoder_h_dim
        ):
            return True
        else:
            return False

    def forward(self, obs_traj, obs_traj_rel, seq_start_end, user_noise=None,mask=None):

        # args1 = parser.parse_args()
        # pred_traj_fake, pred_traj_fake_rel = prediction(args1, obs_traj, obs_traj_rel, seq_start_end)
        # tensor1 = obs_traj
        # tensor2 = pred_traj_fake
        # obs_traj1 = torch.cat((tensor1, tensor2), dim=0)
        # num_frames, num_vehicles, num_features = obs_traj1.shape
        # # 指定文件路径
        # file_path = "data_tensor.txt"
        # # 打开文件进行覆盖写入
        # with open(file_path, 'w') as file:
        #     for frame in range(num_frames):
        #         for vehicle in range(num_vehicles):
        #             vehicle_data = obs_traj1[frame, vehicle, :].cpu()
        #             data_to_save = np.concatenate([[frame, vehicle], vehicle_data])
        #             file.write('\t'.join(map(str, data_to_save)) + '\n')
        # print(f"数据已按帧和车辆编号追加保存到 {file_path}")

        # # 创建一个新的Matplotlib图
        # plt.figure(figsize=(8, 8))
        # # # 可视化观测轨迹
        # plt.plot(obs_traj[:, :, 0], obs_traj[:, :, 1], 'ro-', markersize=8)
        # # 可视化预测轨迹
        # plt.plot(pred_traj_fake[:, :, 0], pred_traj_fake[:, :, 1], 'go-', markersize=8)
        # # 设置图例
        # plt.legend()
        # # 添加轨迹的标签
        # plt.xlabel('X坐标')
        # plt.ylabel('Y坐标')
        # # 显示图形
        # plt.show()


        batch = obs_traj_rel.size(1)  #返回观测序列的数量
        # Encode seq
        final_encoder_h = self.encoder(obs_traj_rel.cuda())  #轨迹相对位置进行编码
        # Pool States
        if self.pooling_type:
            end_pos = obs_traj[-1, :, :]
            pool_h = self.pool_net(final_encoder_h, seq_start_end, end_pos)
            # Construct input hidden states for decoder
            mlp_decoder_context_input = torch.cat(
                [final_encoder_h.view(-1, self.encoder_h_dim), pool_h], dim=1)
        else:
            mlp_decoder_context_input = final_encoder_h.view(
                -1, self.encoder_h_dim)
        # mlp_decoder_context_input是池化后的输出结果

        # Add Noise
        if self.mlp_decoder_needed():
            noise_input = self.mlp_decoder_context(mlp_decoder_context_input)
        else:
            noise_input = mlp_decoder_context_input
        decoder_h = self.add_noise(
            noise_input, seq_start_end, user_noise=user_noise)
        decoder_h = torch.unsqueeze(decoder_h, 0)

        decoder_c = torch.zeros(
            self.num_layers, batch, self.decoder_h_dim
        )

        state_tuple = (decoder_h, decoder_c)
        last_pos = obs_traj[-1]
        last_pos_rel = obs_traj_rel[-1]
        #对输入增加噪音
        # Predict Trajectory
        decoder_out = self.decoder(
            last_pos,
            last_pos_rel,
            state_tuple,
            seq_start_end,
        )
        pred_traj_fake_rel, final_decoder_h = decoder_out

        return pred_traj_fake_rel

class TrajectoryDiscriminator(nn.Module):
    def __init__(
        self, obs_len, pred_len, embedding_dim=64, h_dim=64, mlp_dim=1024,
        num_layers=1, activation='relu', batch_norm=True, dropout=0.0,
        d_type='local'
    ):
        super(TrajectoryDiscriminator, self).__init__()

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len + pred_len
#         print(self.seq_len)
#         exit()
        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.d_type = d_type

        self.encoder1 = Encoder1(
            embedding_dim=embedding_dim,
            h_dim=h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        real_classifier_dims = [h_dim, mlp_dim, 1]
        self.real_classifier = make_mlp(
            real_classifier_dims,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout
        )
        if d_type == 'global':
            mlp_pool_dims = [h_dim + embedding_dim, mlp_dim, h_dim]
            self.pool_net = PoolHiddenNet(
                embedding_dim=embedding_dim,
                h_dim=h_dim,
                mlp_dim=mlp_pool_dims,
                bottleneck_dim=h_dim,
                activation=activation,
                batch_norm=batch_norm
            )

    def forward(self, traj, traj_rel, seq_start_end=None):

        final_h = self.encoder1(traj_rel)

        if self.d_type == 'local':
            classifier_input = final_h.squeeze()
        else:
            classifier_input = self.pool_net(
                final_h.squeeze(), seq_start_end, traj[0]
            )
        scores = self.real_classifier(classifier_input)
        return scores
