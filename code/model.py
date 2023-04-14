import torch
from code.data_loader import BasicDataset
from torch import nn
import numpy as np
import json
import math
import torch.nn.functional as F
import time
import code.utils as utils
import copy


class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()

    def getUsersRating(self, users):
        raise NotImplementedError


class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()

    def bpr_loss(self, users, pos, neg):
        """
        Parameters:
            users: users list
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError


class LightGCN(BasicModel):
    def __init__(self,
                 args,
                 dataset):
        super(LightGCN, self).__init__()
        self.args = args
        self.dataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.user_aspect_dic = self.dataset.user_aspect_dic  # copy.deepcopy
        self.item_aspect_dic = self.dataset.item_aspect_dic  # copy.deepcopy
        self.all_user = self.dataset.all_user
        self.all_item = self.dataset.all_item
        self.aspect_emb_384 = self.dataset.aspect_emb  # [aspect, emb_384]  copy.deepcopy
        self.aspect_emb_64 = dict()  # 转换维度后的aspect嵌入
        self.user_aspect_emb = dict()
        self.item_aspect_emb = dict()
        self.user_padding_aspect = []  # padding后的aspect表示
        self.item_padding_aspect = []
        self.latent_dim = self.args.recdim
        self.n_layers = self.args.layer
        self.keep_prob = self.args.keepprob
        self.A_split = self.args.A_split
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
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        print(f"lgn is already to go(dropout:{self.args.dropout})")

        self.aspect_MLP = nn.Linear(in_features=384, out_features=64, bias=True)  # 统一将所有的aspect的维度进行转换

        # 初始化
        torch.nn.init.xavier_uniform_(self.aspect_MLP.weight)
        torch.nn.init.constant_(self.aspect_MLP.bias, 0)

        self.attention = nn.MultiheadAttention(embed_dim=self.latent_dim, num_heads=1, batch_first=True)

    def aspect_init(self):
        # deepcopy_aspect_emb384 = copy.deepcopy(self.aspect_emb_384) # 在进入之前进行深拷贝
        self.get_ft_aspect_emb(self.aspect_emb_384)
        self.user_aspect_emb, self.item_aspect_emb = self.getAspectEmbedding()
        self.aspect_padding()

        # print("save_txt")

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph

    def computer(self):
        """
        propagate methods for lightGCN
        """
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        if self.args.dropout:
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph
        else:
            g_droped = self.Graph

        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
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

        # 将user/item嵌入传进去计算attention weight
        # 0.6s左右
        user_aspect_emb = self.get_user_aspect_embedding(all_users, users)
        # 0.6s左右
        pos_item_aspect_emb = self.get_item_aspect_embedding(all_items, pos_items)
        # 0.4s左右
        neg_item_aspect_emb = self.get_item_aspect_embedding(all_items, neg_items)

        users_emb = users_emb + user_aspect_emb
        pos_emb = pos_emb + pos_item_aspect_emb
        neg_emb = neg_emb + neg_item_aspect_emb

        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def getUsersRating(self, users, items):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items

        user_aspect_emb = self.get_user_aspect_embedding(all_users, users)  # user已经是按顺序的
        item_aspect_emb = self.get_item_aspect_embedding(all_items, items)  # item已经是按顺序的

        users_emb = users_emb + user_aspect_emb
        items_emb = items_emb + item_aspect_emb

        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    # 如果去掉了MLP，其实这个函数可以简化掉，目前先不简化，看移动MLP之后的效果
    def get_ft_aspect_emb(self, aspect_emb_384):
        # 将所有的aspect从384维转化为64维
        new_emb_384_list = []
        for emb_384 in aspect_emb_384.values():
            new_emb_384 = torch.tensor(np.array(emb_384)).to(torch.float32).to(self.args.device)
            new_emb_384_list.append(new_emb_384)

        new_emb_384_tensor = torch.stack(new_emb_384_list)
        print("new_emb_384_tensor.shape:", new_emb_384_tensor.shape)
        emb_64 = self.aspect_MLP(new_emb_384_tensor)

        self.aspect_emb_64 = dict(zip(list(aspect_emb_384.keys()), emb_64.tolist()))

    def getAspectEmbedding(self):
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

    # 在前面  不费时间
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

    # 计算每个batch中user对应的aspect embedding列表
    def get_user_aspect_embedding(self, user_emb, users):
        # 主要是遍历的时间
        batch_pad_aspect = torch.index_select(self.user_padding_aspect, 0, users)
        batch_user_emb = torch.index_select(user_emb, 0, users)

        batch_user_emb = torch.unsqueeze(batch_user_emb, 1)  # 扩展维度到3维

        # 0.0006s的时间
        aspect_emb_user, _ = self.attention(batch_user_emb, batch_pad_aspect, batch_pad_aspect)
        aspect_emb_user = torch.squeeze(aspect_emb_user)

        # return torch.stack(user_aspect_emb).to(self.args.device)
        return aspect_emb_user.to(self.args.device)

    def get_item_aspect_embedding(self, item_emb, items):
        items = torch.tensor(items).to(self.args.device)  # items本来是numpy类型
        batch_pad_aspect = torch.index_select(self.item_padding_aspect, 0, items)
        batch_item_emb = torch.index_select(item_emb, 0, items)

        batch_item_emb = torch.unsqueeze(batch_item_emb, 1)  # 扩展维度到3维

        # 0.0006s的时间
        aspect_emb_item, _ = self.attention(batch_item_emb, batch_pad_aspect, batch_pad_aspect)

        aspect_emb_item = torch.squeeze(aspect_emb_item)

        """item_aspect_emb = []
        # 6s左右 时间很长
        for i in items:
            if i in self.item_aspect_emb.keys():
                aspect_emb_list = self.item_aspect_emb[i].to(self.args.device)
                # 0.002s左右
                aspect_emb_item, _ = self.attention(torch.unsqueeze(item_emb[i], 0), aspect_emb_list, aspect_emb_list)
                aspect_emb_item = torch.squeeze(aspect_emb_item)
                item_aspect_emb.append(aspect_emb_item)
            else:
                item_aspect_emb.append(torch.tensor([0 for i in range(self.args.recdim)]).to(self.args.device))
"""
        # return torch.stack(item_aspect_emb).to(self.args.device)
        return aspect_emb_item.to(self.args.device)

    # query：user/item   key/value: aspect
    def attention_aspectAgg(self, query, key, value, mask=None, dropout=None):
        # query：[batch_size, 1, emb_dim]  user/item
        # key/values: [batch_size, n_aspect, emb_dim]  aspect
        # 首先取query的最后一维的大小，对应user/item/aspect的嵌入维度  比如64
        d_k = query.size(-1)  # 64
        # 按照注意力公式，将query与key的转置相乘，这里面key是将最后两个维度进行转置，再除以缩放系数得到注意力得分张量scores
        # query = torch.unsqueeze(query, 0)  # [emb_dim] -> [1, emb_dim]
        scores = (torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)).to(
            self.args.device)  # [batch_size, 1, n_aspects]

        # 对scores的最后一维进行softmax操作
        p_attn = F.softmax(scores, dim=-1)

        # 之后判断是否使用dropout进行随机置0
        if dropout is not None:
            p_attn = dropout(p_attn)

        # 最后，根据公式将p_attn与value张量相乘获得最终的query注意力表示，返回aspect的对user/item的贡献值
        # [batch_size, 1, emb_dim]
        return torch.matmul(p_attn, value)

        # 计算嵌入和loss

    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        # loss
        """reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(len(users))"""

        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        return loss  # , reg_loss

    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        # print('forward')
        # all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma