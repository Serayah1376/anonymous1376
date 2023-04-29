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
        self.aspect_emb_384 = self.dataset.aspect_emb  # [aspect, emb_384]
        self.aspect_emb_64 = dict()
        self.user_aspect_emb = dict()
        self.item_aspect_emb = dict()
        self.user_padding_aspect = []  # user/item aspect padding
        self.item_padding_aspect = []

        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()  # dgl.heterograph
        print(f"lgn is already to go(dropout:{self.args.dropout})")

        self.layers = nn.ModuleList()
        for i in range(self.n_layers):
            self.layers.append(GCNLayer(args))

        self.__init_weight()

    def __init_weight(self):
        # user/item embedding weight init
        self.embedding_user = torch.nn.Parameter(torch.randn(self.num_users, self.latent_dim))
        self.embedding_item = torch.nn.Parameter(torch.randn(self.num_items, self.latent_dim))

        # embedding init
        """if self.args.pretrain == 0:  # true
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            print('use NORMAL distribution initilizer')
        else:
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            print('use pretarined data')"""

        # convert the dim of aspect
        self.aspect_MLP = nn.Linear(in_features=self.args.aspect_LMdim, out_features=self.args.recdim, bias=True)

        self.attention = nn.MultiheadAttention(embed_dim=self.latent_dim, num_heads=self.head_num,
                                               batch_first=True)  # head_num

        nn.init.xavier_normal_(self.aspect_MLP.weight)
        nn.init.constant_(self.aspect_MLP.bias, 0)

    # aspect pre-train weight
    def aspect_init(self):
        self.get_ft_aspect_emb(self.aspect_emb_384)
        self.user_aspect_emb, self.item_aspect_emb = self.get_Pretrain_Aspect()
        self.aspect_padding()

    def computer(self):
        user_embed = [self.embedding_user]
        item_embed = [self.embedding_item]
        h = {'user': self.embedding_user, 'item': self.embedding_item}
        for layer in self.layers:
            h_item = layer(self.Graph, h, ('user', 'rate', 'item'))
            h_user = layer(self.Graph, h, ('item', 'rated by', 'user'))

            user_aspect_emb = self.get_aspect_embedding(h_user, torch.tensor(self.all_user).to(self.args.device),
                                                        is_user=True)
            item_aspect_emb = self.get_aspect_embedding(h_item, torch.tensor(self.all_item).to(self.args.device),
                                                        is_user=False)

            h_user = (h_user + user_aspect_emb) / 2
            h_item = (h_item + item_aspect_emb) / 2

            h = {'user': h_user, 'item': h_item}

            user_embed.append(h_user)
            item_embed.append(h_item)

        user_embed = torch.stack(user_embed, dim=1)
        users = torch.mean(user_embed, dim=1)

        item_embed = torch.stack(item_embed, dim=1)
        items = torch.mean(item_embed, dim=1)

        """all_aspect_emb = torch.stack(all_aspect_emb, dim=1)
        aspect_out = torch.mean(all_aspect_emb, dim=1)
        user_aspect, item_aspect = torch.split(aspect_out, [self.num_users, self.num_items])"""

        return users, items

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]

        users_emb_ego = self.embedding_user[users]
        pos_emb_ego = self.embedding_item[pos_items]
        neg_emb_ego = self.embedding_item[neg_items]
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def getUsersRating(self, users, items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items

        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def get_ft_aspect_emb(self, aspect_emb_384):
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
        aspect_emb, _ = self.attention(batch_emb, batch_pad_aspect, batch_pad_aspect)
        aspect_emb = torch.squeeze(aspect_emb)

        if torch.count_nonzero(aspect_emb) == 0:
            aspect_emb = emb  # self

        return aspect_emb.to(self.args.device)

    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        # loss
        # regloss1
        """reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(len(users))"""

        """pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))"""  # bpr loss

        return users_emb, pos_emb, neg_emb  # , reg_loss # , loss


class GCNLayer(nn.Module):
    def __init__(self, args):
        super(GCNLayer, self).__init__()
        self.args = args
        self.sigma = args.sigma
        self.gamma = args.gamma2
        self.k = args.k  # number of aggregated neighbors

    def similarity_matrix(self, X, sigma=1.0, gamma=2.0):
        dists = torch.cdist(X, X)  # formula 7
        # formula 8
        sims = torch.exp(-dists / (sigma * dists.mean(dim=-1).mean(dim=-1).reshape(-1, 1, 1)))
        return sims

    # submodular
    def submodular_selection_feature(self, nodes):
        device = nodes.mailbox['m'].device
        feature = nodes.mailbox['m']
        sims = self.similarity_matrix(feature, self.sigma, self.gamma)  # 所有节点的相似性计算

        batch_num, neighbor_num, feature_size = feature.shape
        nodes_selected = []
        cache = torch.zeros((batch_num, 1, neighbor_num), device=device)

        # use the greedy algorithm to pick k neighbors
        for i in range(self.k):
            gain = torch.sum(torch.maximum(sims, cache) - cache, dim=-1)

            selected = torch.argmax(gain, dim=1)
            cache = torch.maximum(sims[torch.arange(batch_num, device=device), selected].unsqueeze(1), cache)

            nodes_selected.append(selected)

        return torch.stack(nodes_selected).t()

    def sub_reduction(self, nodes):
        # -1 indicate user-> node, which does not include category information
        mail = nodes.mailbox['m']
        batch_size, neighbor_size, feature_size = mail.shape

        if (-1 in nodes.mailbox['c']) or nodes.mailbox['m'].shape[1] <= self.k:
            mail = mail.sum(dim=1)
        else:
            neighbors = self.submodular_selection_feature(nodes)
            mail = mail[torch.arange(batch_size, dtype=torch.long, device=mail.device).unsqueeze(-1), neighbors]
            mail = mail.sum(dim=1)
        return {'h': mail}

    def category_aggregation(self, edges):
        return {'c': edges.src['category'], 'm': edges.src['h']}

    def forward(self, graph, node_f, etype):
        with graph.local_scope():
            src, _, dst = etype  # user / item
            feat_src = node_f[src]
            feat_dst = node_f[dst]

            # D^(-1/2)
            degs = graph.out_degrees(etype=etype).float().clamp(min=1)
            norm = torch.pow(degs, -0.5)
            shp = norm.shape + (1,) * (feat_src.dim() - 1)  # feat_src.dim(): 2
            norm = torch.reshape(norm, shp)
            feat_src = feat_src * norm

            graph.nodes[src].data['h'] = feat_src
            graph.update_all(self.category_aggregation, self.sub_reduction, etype=etype)
            # graph.update_all(message_func=fn.copy_u('n_f', 'm'), reduce_func=fn.sum('m', 'n_f'))

            # norm
            rst = graph.nodes[dst].data['h']
            degs = graph.in_degrees(etype=etype).float().clamp(min=1)
            norm = torch.pow(degs, -0.5)
            shp = norm.shape + (1,) * (feat_dst.dim() - 1)
            norm = torch.reshape(norm, shp)
            rst = rst * norm

            return rst