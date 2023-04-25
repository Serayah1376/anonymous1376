import torch
from code.data_loader import BasicDataset
from torch import nn
import numpy as np
import json
import math
import torch.nn.functional as F
import time
import code.utils as utils
import dgl.function as fn
import copy


class Model(nn.Module):
    def __init__(self,
                 args,
                 dataset):
        super(Model, self).__init__()
        self.args = args
        self.dataset = dataset
        self.latent_dim = self.args.recdim
        self.n_layers = self.args.layer
        self.keep_prob = self.args.keepprob
        self.A_split = self.args.A_split
        self.head_num = self.args.head_num
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.all_user = self.dataset.all_user
        self.all_item = self.dataset.all_item

        # aspect init
        self.user_aspect_dic = self.dataset.user_aspect_dic
        self.item_aspect_dic = self.dataset.item_aspect_dic
        self.aspect_emb_384 = self.dataset.aspect_emb  # [aspect, emb_384]  copy.deepcopy
        self.aspect_emb_64 = dict()  # 转换维度后的aspect嵌入
        self.user_aspect_emb = dict()
        self.item_aspect_emb = dict()
        self.user_padding_aspect = []  # padding后的aspect表示
        self.item_padding_aspect = []

        self.layers = nn.ModuleList()
        for i in range(self.n_layers):
            self.layers.append(GCNLayer())

        self.__init_weight()

    def __init_weight(self):
        # user/item embedding weight init
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        if self.args.pretrain == 0:  # true
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            print('use NORMAL distribution initilizer')
        else:
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            print('use pretarined data')

        # model weight init
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()  # 交互稀疏矩阵 dgl.graph生成
        print(f"lgn is already to go(dropout:{self.args.dropout})")

        self.aspect_MLP = nn.Linear(in_features=384, out_features=64, bias=True)  # 统一将所有的aspect的维度进行转换

        self.attention = nn.MultiheadAttention(embed_dim=self.latent_dim, num_heads=self.head_num,
                                               batch_first=True)  # head_num

        torch.nn.init.xavier_uniform_(self.aspect_MLP.weight)
        torch.nn.init.constant_(self.aspect_MLP.bias, 0)
        torch.nn.init.xavier_uniform_(self.attention.in_proj_weight)
        torch.nn.init.constant_(self.attention.in_proj_bias, 0)
        torch.nn.init.xavier_uniform_(self.attention.out_proj.weight)
        torch.nn.init.constant_(self.attention.out_proj.bias, 0)

    # aspect pre-train weight
    def aspect_init(self):
        # deepcopy_aspect_emb384 = copy.deepcopy(self.aspect_emb_384) # 在进入之前进行深拷贝
        self.get_ft_aspect_emb(self.aspect_emb_384)
        self.user_aspect_emb, self.item_aspect_emb = self.get_Pretrain_Aspect()
        self.aspect_padding()

    def computer(self):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        for layer in self.layers:
            all_emb = layer(self.Graph, all_emb)

            new_item_list = self.all_item + len(self.all_user)
            user_aspect_emb = self.get_aspect_embedding(all_emb[self.all_user],
                                                        torch.tensor(self.all_user).to(self.args.device), is_user=True)
            item_aspect_emb = self.get_aspect_embedding(all_emb[new_item_list],
                                                        torch.tensor(self.all_item).to(self.args.device), is_user=False)
            all_emb = (all_emb + torch.cat([user_aspect_emb, item_aspect_emb])) / 2

            embs.append(all_emb)

        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])

        return users, items

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]

        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def getUsersRating(self, users, items):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items

        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    # 转化aspect维度
    def get_ft_aspect_emb(self, aspect_emb_384):
        # 将所有的aspect从384维转化为64维
        new_emb_384_list = []
        for emb_384 in aspect_emb_384.values():
            new_emb_384 = torch.tensor(np.array(emb_384)).to(torch.float32).to(self.args.device)
            new_emb_384_list.append(new_emb_384)

        new_emb_384_tensor = torch.stack(new_emb_384_list)
        emb_64 = self.aspect_MLP(new_emb_384_tensor)
        self.aspect_emb_64 = dict(zip(list(aspect_emb_384.keys()), emb_64.tolist()))

    def get_Pretrain_Aspect(self):
        user_aspect_embedding = dict()
        item_aspect_embedding = dict()
        for i in self.all_user:  # i: torch.tensor
            i = int(i)
            if i in self.user_aspect_dic.keys():
                a_vector_list = []
                for a in self.user_aspect_dic[i]:
                    a_vector_list.append(torch.tensor(self.aspect_emb_64[a]))
                if len(a_vector_list) != 0:
                    vector_tensor = torch.stack(a_vector_list)  # [n_aspects, 64]
                    user_aspect_embedding[i] = vector_tensor

        for i in self.all_item:  # i: torch.tensor
            i = int(i)
            if i in self.item_aspect_dic.keys():
                a_vector_list = []
                for a in self.item_aspect_dic[i]:
                    a_vector_list.append(torch.tensor(self.aspect_emb_64[a]))  # user 和 item 的aspect并不完全一样
                if len(a_vector_list) != 0:
                    vector_tensor = torch.stack(a_vector_list)
                    item_aspect_embedding[i] = vector_tensor

        return user_aspect_embedding, item_aspect_embedding

    def aspect_padding(self):  # entity: user/item
        for i in self.all_user:
            if i in self.user_aspect_emb.keys():
                aspect_emb_list = self.user_aspect_emb[i].to(self.args.device)  # tensor
                padding_aspect = utils.aspect_padding(aspect_emb_list, self.args.max_len)
            else:
                padding_aspect = utils.aspect_padding(None, self.args.max_len)

            self.user_padding_aspect.append(padding_aspect)

        for i in self.all_item:
            i = int(i)
            if i in self.item_aspect_emb.keys():
                aspect_emb_list = self.item_aspect_emb[i].to(self.args.device)  # tensor
                padding_aspect = utils.aspect_padding(aspect_emb_list, self.args.max_len)
            else:
                padding_aspect = utils.aspect_padding(None, self.args.max_len)

            self.item_padding_aspect.append(padding_aspect)  # 按照顺序的

        self.user_padding_aspect = torch.stack(self.user_padding_aspect).to(self.args.device)
        self.item_padding_aspect = torch.stack(self.item_padding_aspect).to(self.args.device)

    # 计算user/item对应的aspect embedding列表
    def get_aspect_embedding(self, emb, index, is_user=True):
        if is_user:
            batch_pad_aspect = torch.index_select(self.user_padding_aspect, 0, index)
        else:
            batch_pad_aspect = torch.index_select(self.item_padding_aspect, 0, index)

        batch_emb = torch.index_select(emb, 0, index)
        batch_emb = torch.unsqueeze(batch_emb, 1)
        aspect_emb_item, _ = self.attention(batch_emb, batch_pad_aspect, batch_pad_aspect)
        aspect_emb_item = torch.squeeze(aspect_emb_item)

        return aspect_emb_item.to(self.args.device)

    # 计算嵌入和loss
    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        # loss
        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(len(users))

        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        return users_emb, pos_emb, neg_emb, reg_loss, loss


# 基于DGL框架的lightGCN，GraphConv不是lightGCN
class GCNLayer(nn.Module):
    def __init__(self):
        super(GCNLayer, self).__init__()

    def forward(self, graph, node_f):  # graph： dgl.graph生成的
        with graph.local_scope():  # 不改变graph中的值
            # D^-1/2
            degs = graph.out_degrees().to(node_f.device).float().clamp(min=1)  # 出度
            norm = torch.pow(degs, -0.5).view(-1, 1)

            node_f = node_f * norm  # 归一化

            graph.ndata['n_f'] = node_f  # 特征赋值

            # 更新，可以选择更新的方式
            graph.update_all(message_func=fn.copy_u('n_f', 'm'), reduce_func=fn.sum('m', 'n_f'))

            rst = graph.ndata['n_f']

            degs = graph.in_degrees().to(node_f.device).float().clamp(min=1)  # 入度
            norm = torch.pow(degs, -0.5).view(-1, 1)
            rst = rst * norm

            return rst