import torch
from code.data_loader import BasicDataset
from torch import nn
import numpy as np
import json
from gensim.models import Word2Vec


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
        self.user_aspect_dic = self.dataset.user_aspect_dic
        self.item_aspect_dic = self.dataset.item_aspect_dic
        self.all_user = self.dataset.all_user
        self.all_item = self.dataset.all_item
        self.aspect_emb_384 = self.dataset.aspect_emb  # [aspect, emb_384]
        self.aspect_emb_64 = dict()  # 转换维度后的aspect嵌入
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

        # self.W1 = nn.Parameter(torch.ones(1, requires_grad=True))
        # self.W2 = nn.Parameter(torch.ones(1, requires_grad=True))

        """self.aspect_MLP_u = nn.Linear(in_features=64, out_features=384, bias=True)  
        self.aspect_MLP_i = nn.Linear(in_features=64, out_features=384, bias=True) """

        self.aspect_MLP = nn.Linear(in_features=384, out_features=64, bias=True)  # 统一将所有的aspect的维度进行转换

        """self.aspect_MLP_user = nn.Linear(in_features=384, out_features=64, bias=True)  
        self.aspect_MLP_item = nn.Linear(in_features=384, out_features=64, bias=True) """

        # 初始化
        torch.nn.init.xavier_uniform_(self.aspect_MLP.weight)
        torch.nn.init.constant_(self.aspect_MLP.bias, 0)

        """torch.nn.init.xavier_uniform_(self.aspect_MLP_user.weight)
        torch.nn.init.constant_(self.aspect_MLP_user.bias, 0)
        torch.nn.init.xavier_uniform_(self.aspect_MLP_item.weight)
        torch.nn.init.constant_(self.aspect_MLP_item.bias, 0)"""

        self.dropout = nn.Dropout(0.1)

        self.get_ft_aspect_emb(self.aspect_emb_384)
        self.user_aspect_emb, self.item_aspect_emb = self.getAspectEmbedding()

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
        # print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        # new_users = self.dropout(self.aspect_MLP_u(users))
        # new_items = self.dropout(self.aspect_MLP_i(items))
        return users, items

    def getUsersRating(self, users, items):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items

        user_aspect_emb = self.get_user_aspect_embedding(users)
        item_aspect_emb = self.get_item_aspect_embedding(items)  # item已经是按顺序的

        users_emb = users_emb + user_aspect_emb
        items_emb = items_emb + item_aspect_emb

        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    # 如果去掉了MLP，其实这个函数可以简化掉，目前先不简化，看移动MLP之后的效果
    def get_ft_aspect_emb(self, aspect_emb_384):
        # 将所有的aspect从384维转化为64维
        for aspect, emb_384 in aspect_emb_384.items():
            self.aspect_emb_64[aspect] = self.aspect_MLP(torch.tensor(np.array(emb_384)).to(torch.float32))

        """# 不改变维度
        for k, v in user_aspect_emb.items():
            self.user_aspect_emb[k] = torch.tensor(np.array(v)).to(torch.float32)

        for k, v in item_aspect_emb.items():
            self.item_aspect_emb[k] = torch.tensor(np.array(v)).to(torch.float32)

        # 改变维度
        u_key = []
        u_emb = []
        for k, v in user_aspect_emb.items():
            u_key.append(k)
            u_emb.append(v)
            #self.user_aspect_emb[k] = v # torch.tensor(np.array(v)).to(torch.float32)

        # new_ua_emb = self.aspect_MLP(torch.tensor(np.array(u_emb)).to(torch.float32))

        for k, v in zip(u_key, u_emb):
            self.user_aspect_emb[k] = v

        i_key = []
        i_emb = []
        for k, v in item_aspect_emb.items():
            i_key.append(k)
            i_emb.append(v)
            #self.item_aspect_emb[k] = v# torch.tensor(np.array(v)).to(torch.float32)

        # new_ia_emb = self.aspect_MLP(torch.tensor(np.array(i_emb)).to(torch.float32))

        for k, v in zip(i_key, i_emb):
            self.item_aspect_emb[k] = v"""

    def getAspectEmbedding(self):
        user_aspect_embedding = dict()
        item_aspect_embedding = dict()
        for i in self.all_user:  # i: torch.tensor
            i = int(i)
            if i in self.user_aspect_dic.keys():
                a_vector_list = []
                for a in self.user_aspect_dic[i]:
                    a_vector_list.append(self.aspect_emb_64[a])
                if len(a_vector_list) != 0:
                    vector_tensor = torch.stack(a_vector_list)  # [n_aspects, 384]
                    # new_vector_tensor = self.aspect_MLP_user(vector_tensor)  # [n_aspects, 64]
                    new_vector_tensor = torch.mean(vector_tensor, axis=0)  # 求平均)
                    """a_vector_list = np.array(a_vector_list)
                    a_vector_list = np.average(a_vector_list, axis=0)  # 均值"""
                    user_aspect_embedding[i] = new_vector_tensor

        for i in self.all_item:  # i: torch.tensor
            i = int(i)
            if i in self.item_aspect_dic.keys():
                a_vector_list = []
                for a in self.item_aspect_dic[i]:
                    a_vector_list.append(self.aspect_emb_64[a])  # user 和 item 的aspect并不完全一样
                if len(a_vector_list) != 0:
                    vector_tensor = torch.stack(a_vector_list)
                    # new_vector_tensor = self.aspect_MLP_item(vector_tensor)  # [n_aspects, dim] 384->64
                    new_vector_tensor = torch.mean(vector_tensor, axis=0)
                    """a_vector_list = np.array(a_vector_list)
                    a_vector_list = np.average(a_vector_list, axis=0)"""
                    item_aspect_embedding[i] = new_vector_tensor

        return user_aspect_embedding, item_aspect_embedding

    # 计算aspect的嵌入
    def get_user_aspect_embedding(self, users):
        user_aspect_emb = []
        for i in users:
            i = int(i)
            if i in self.user_aspect_emb.keys():
                user_aspect_emb.append(self.user_aspect_emb[i].tolist())
            else:
                user_aspect_emb.append([0 for _ in range(64)])  # aspect嵌入维度，可变！！！！

        return torch.tensor(user_aspect_emb).to(self.args.device)

    def get_item_aspect_embedding(self, items):
        item_aspect_emb = []
        for i in items:
            i = int(i)
            if i in self.item_aspect_emb.keys():
                item_aspect_emb.append(self.item_aspect_emb[i].tolist())
            else:
                item_aspect_emb.append([0 for _ in range(64)])
        return torch.tensor(item_aspect_emb).to(self.args.device)

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]

        user_aspect_emb = self.get_user_aspect_embedding(users)
        pos_item_aspect_emb = self.get_item_aspect_embedding(pos_items)
        neg_item_aspect_emb = self.get_item_aspect_embedding(neg_items)

        users_emb = users_emb + user_aspect_emb
        pos_emb = pos_emb + pos_item_aspect_emb
        neg_emb = neg_emb + neg_item_aspect_emb

        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

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

        return loss, reg_loss

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
