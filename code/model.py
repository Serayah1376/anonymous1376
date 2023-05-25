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
        self.num_aspects = self.dataset.n_aspect
        self.all_user = self.dataset.all_user
        self.all_item = self.dataset.all_item
        self.new_λ = self.args.new_λ

        # aspect init
        self.user_aspect_dic = self.dataset.user_aspect_dic  # [userID, aspect]
        self.item_aspect_dic = self.dataset.item_aspect_dic  # [itemID, aspect]
        # trainable aspect
        self.user_aspect_ID_dic = self.dataset.user_aspect_ID_dic  # [userID, aspectID]
        self.item_aspect_ID_dic = self.dataset.item_aspect_ID_dic  # [itemID, aspectID]

        self.aspect_emb_384 = self.dataset.aspect_emb  # [aspect, emb_384]
        self.UserAspectNet = self.dataset.UserAspectNet  # user, aspect  bipartite graph
        self.aspect_emb_64 = dict()
        self.user_aspect_emb = dict()
        self.item_aspect_emb = dict()
        self.user_padding_aspect = []  # user/item aspect padding
        self.item_padding_aspect = []
        self.new_λ = self.args.new_λ
        self.train_add_newAspect = self.args.train_add_newAspect
        self.test_add_newAspect = self.args.test_add_newAspect

        # aspect mask: 在选择新的aspect时，屏蔽到已经交互过的aspect
        aspect_mask_pre = torch.tensor(self.UserAspectNet.toarray())
        self.aspect_mask = torch.where(aspect_mask_pre == 0, aspect_mask_pre, float('-inf')).to(
            self.args.device)  # 交互过的地方均为负无穷

        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()  # dgl.heterograph
        # self.Graph = self.dataset.getAspectSparseGraph()  # 含有aspect的异构图
        print(f"lgn is already to go(dropout:{self.args.dropout})")

        self.layers = nn.ModuleList()
        for i in range(self.n_layers):
            self.layers.append(GCNLayer(args))

        self.__init_weight()

    def __init_weight(self):
        # user/item embedding weight init
        self.embedding_user = torch.nn.Parameter(torch.randn(self.num_users, self.latent_dim))
        self.embedding_item = torch.nn.Parameter(torch.randn(self.num_items, self.latent_dim))

        # make aspect embedding changable
        self.embedding_aspect = torch.nn.Parameter(torch.randn(self.num_aspects, self.latent_dim))

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
        nn.init.xavier_normal_(self.aspect_MLP.weight)
        nn.init.constant_(self.aspect_MLP.bias, 0)

        # for aspect_conditional_encoder
        self.MLP = nn.Linear(in_features=self.args.recdim * 2, out_features=self.args.recdim, bias=True)
        nn.init.xavier_normal_(self.MLP.weight)
        nn.init.constant_(self.MLP.bias, 0)
        relu = nn.ReLU(inplace=True)
        # LeakyReLU = nn.LeakyReLU(negative_slope=5e-2)  搜一下看怎么设置

        self.attention = nn.MultiheadAttention(embed_dim=self.latent_dim, num_heads=self.head_num,
                                               batch_first=True)  # head_num

    # aspect pre-train weight
    def aspect_init(self):
        self.get_ft_aspect_emb(self.aspect_emb_384)
        self.user_aspect_emb, self.item_aspect_emb = self.get_Pretrain_Aspect()
        # self.user_aspect_emb, self.item_aspect_emb = self.get_trainable_Aspect()  # trainable aspect 没有建模在图中
        self.new_aspect_emb = self.candidate_aspect()  # 得到需要新加入的aspect，是预处理好的 torch.Size([batch_size, num_new_aspects, emb])
        self.aspect_padding()  # 使得每个user和item的aspect数相等，aspect数取max_len个，目前是64

    def computer(self, is_train=True):
        user_embed = [self.embedding_user]
        item_embed = [self.embedding_item]
        h = {'user': self.embedding_user, 'item': self.embedding_item}
        # h = {'user':  self.embedding_user, 'item': self.embedding_item, 'aspect': self.embedding_aspect}

        # 生成噪声  XSimGCL 在每一层上加的噪声
        random_noise_user = torch.rand_like(self.embedding_user).cuda()
        random_noise_item = torch.rand_like(self.embedding_item).cuda()
        user_embed_cl = self.embedding_user
        item_embed_cl = self.embedding_item

        h_user = self.embedding_user
        h_item = self.embedding_item
        for layer in self.layers:
            h_item = layer(self.Graph, h, ('user', 'rate', 'item'))
            h_user = layer(self.Graph, h, ('item', 'rated by', 'user'))

            user_aspect_emb = self.get_aspect_embedding(h_user, torch.tensor(self.all_user).to(self.args.device),
                                                        is_user=True)
            item_aspect_emb = self.get_aspect_embedding(h_item, torch.tensor(self.all_item).to(self.args.device),
                                                        is_user=False)

            h_user = (h_user + user_aspect_emb) / 2
            h_item = (h_item + item_aspect_emb) / 2

            # 加上新的aspect
            # h_user = self.aspect_diversity(h_user, torch.tensor(self.all_user).to(self.args.device))

            # add for aspect graph
            """h_item_aspect = layer(self.Graph, h, ('aspect', 'mentioned by ai', 'item'))
            h_user_aspect = layer(self.Graph, h, ('aspect', 'mentioned by au', 'user'))
            h_user = h_user + h_user_aspect
            h_item = h_item + h_item_aspect"""

            # 每一层加上噪声
            if self.args.perturbed and is_train:  # 预测的时候不加噪声
                print(torch.sign(h_user) * F.normalize(random_noise_user, dim=-1) * self.args.eps)  # self.eps待定
                # h_item += torch.sign(h_item) * F.normalize(random_noise_item, dim=-1) * self.args.eps   # self.eps待定"""

            if layer == self.args.layer_cl - 1:  # XSimGCL 中 选出其中一层来对比
                user_embed_cl = h_user

            h = {'user': h_user, 'item': h_item}
            # h = {'user': h_user, 'item': h_item, 'aspect': self.embedding_aspect}  # aspect 的嵌入虽然在layer中没有更新，但是自己更新了

            user_embed.append(h_user)
            item_embed.append(h_item)

        user_embed = torch.stack(user_embed, dim=1)
        user_embed = torch.mean(user_embed, dim=1)

        item_embed = torch.stack(item_embed, dim=1)
        item_embed = torch.mean(item_embed, dim=1)

        return user_embed, item_embed, user_embed_cl, item_embed_cl

    def getEmbedding(self, users, pos_items, neg_items):  # users/pos_items: tensor
        all_users, all_items, user_embed_cl, item_embed_cl = self.computer()  # 只传入本次需要计算的即可

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]

        if self.train_add_newAspect:
            # users_emb_diversity = self.aspect_diversity(users_emb, users)  # 只对该批次user加diversity, 得到加入new aspect的user emb
            users_emb_diversity = self.aspect_diversity2(users_emb, users)  # 使用注意力机制进行新aspect的筛选

        """users_emb_ego = self.embedding_user[users]
        pos_emb_ego = self.embedding_item[pos_items]
        neg_emb_ego = self.embedding_item[neg_items]"""
        # return users_emb, pos_emb, neg_emb, all_users, all_items, user_embed_cl, item_embed_cl
        return users_emb_diversity, pos_emb, neg_emb, all_users, all_items, users_emb  # user_embed_cl[users]# user_embed_cl, item_embed_cl

    def getUsersRating(self, users, items):
        all_users, all_items, _, _ = self.computer(is_train=False)
        users_emb = all_users[users]

        # 测试期间加不加需要测试一下
        if self.test_add_newAspect:
            # users_emb = self.aspect_diversity(users_emb, users)  # add new aspects  测试的时候不加了，直接用已经训练好的user_emb
            users_emb = self.aspect_diversity2(users_emb, users)  # 使用注意力机制进行新aspect的筛选

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
        user_mean_aspect = []
        for i in self.all_user:  # i: torch.tensor
            i = int(i)
            if i in self.user_aspect_dic.keys():
                a_vector_list = []
                for a in self.user_aspect_dic[i]:
                    a_vector_list.append(torch.tensor(self.aspect_emb_64[a]))
                if len(a_vector_list) != 0:
                    vector_tensor = torch.stack(a_vector_list)  # [n_aspects, 64]
                    user_aspect_embedding[i] = vector_tensor
                    user_mean_aspect.append(torch.mean(torch.stack(a_vector_list), dim=0))  # 所有aspect的均值
            else:
                user_mean_aspect.append(torch.zeros(self.args.recdim))  # 没有aspect的情况
        self.user_mean_aspect = torch.stack(user_mean_aspect).to(self.args.device)  # 每个user交互过的aspect emb的均值

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

    # 计算user的aspect表示与未交互的aspect之间的相似度，从而选择出top-k个aspect
    def candidate_aspect(self):
        aspect_emb = torch.tensor(list(self.aspect_emb_64.values())).to(self.args.device)  # 所有aspect嵌入
        dicts = self.cos_sim(self.user_mean_aspect, aspect_emb)  # 计算user的表示与其他所有aspect表示的相似度 [num_user, num_aspects]
        masked_dicts = self.aspect_mask + dicts
        _, sorted_index = torch.sort(masked_dicts, dim=1, descending=True)
        sorted_index = sorted_index[:, : self.args.new_aspects]  # [num_user, num_aspects]
        new_aspect_emb = []
        for i in range(len(sorted_index)):
            new_aspect_emb.append(torch.index_select(aspect_emb, 0, sorted_index[i]))  # 每个user对应的new aspect

        new_aspect_emb = torch.stack(new_aspect_emb)  # torch.Size([batch_size, num_new_aspects, emb])
        return new_aspect_emb

    def get_trainable_Aspect(self):
        user_aspect_embedding = dict()
        item_aspect_embedding = dict()
        for i in self.all_user:  # i: torch.tensor
            i = int(i)
            if i in self.user_aspect_ID_dic.keys():
                a_vector_list = []
                for a in self.user_aspect_ID_dic[i]:  # ID
                    a_vector_list.append(torch.tensor(self.embedding_aspect[a]))
                if len(a_vector_list) != 0:
                    vector_tensor = torch.stack(a_vector_list)  # [n_aspects, 64]
                    user_aspect_embedding[i] = vector_tensor

        for i in self.all_item:  # i: torch.tensor
            i = int(i)
            if i in self.item_aspect_ID_dic.keys():
                a_vector_list = []
                for a in self.item_aspect_ID_dic[i]:
                    a_vector_list.append(torch.tensor(self.embedding_aspect[a]))  # user 和 item 的aspect并不完全一样
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

        # [all_user_num, max_len, 64] 其中max_len是每个user/item列表的最大长度
        self.user_padding_aspect = torch.stack(self.user_padding_aspect).to(self.args.device)
        self.item_padding_aspect = torch.stack(self.item_padding_aspect).to(self.args.device)

    # 计算user/item对应的aspect embedding列表
    def get_aspect_embedding(self, emb, index, is_user=True):
        if is_user:
            batch_pad_aspect = torch.index_select(self.user_padding_aspect, 0, index)  # [batch_size, max_len, 64]
        else:
            batch_pad_aspect = torch.index_select(self.item_padding_aspect, 0, index)

        batch_emb = torch.index_select(emb, 0, index)
        batch_emb = torch.unsqueeze(batch_emb, 1)
        aspect_emb, _ = self.attention(batch_emb, batch_pad_aspect, batch_pad_aspect)  # Q, K, V
        aspect_emb = torch.squeeze(aspect_emb)

        if torch.count_nonzero(aspect_emb) == 0:  # 没有aspect的情况
            aspect_emb = emb  # self

        return aspect_emb.to(self.args.device)

    def bpr_loss(self, users, pos, neg, aspect_emb):
        """(users_emb, pos_emb, neg_emb,
         all_users, all_items,
         user_embed_cl, item_embed_cl) = self.getEmbedding(users.long(), pos.long(), neg.long())"""

        (users_emb, pos_emb, neg_emb,
         all_users, all_items,
         users_emb_orginal) = self.getEmbedding(users.long(), pos.long(), neg.long())

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
        # cl_loss = self.args.cl_rate * self.cal_cl_loss([users,pos], users_emb, users_emb_orginal)

        # 在aspect的影响下计算
        # u | aspect, i | aspect   for
        users_emb = self.Aspect_condition_encoder(users_emb, aspect_emb)  # aspect alignment
        pos_emb = self.Aspect_condition_encoder(pos_emb, aspect_emb)

        return users_emb, pos_emb, neg_emb  # , cl_loss #, reg_loss # , loss

    # 计算对比损失
    def cal_cl_loss(self, idx, user_view1, user_view2):
        u_idx = torch.unique(idx[0])
        user_cl_loss = self.InfoNCE(user_view1, user_view2, self.args.temp)  # 只有一个超参数temp
        return user_cl_loss  # + item_cl_loss

    def cal_cl_loss_before(self, idx, user_view1, user_view2, item_view1, item_view2):
        u_idx = torch.unique(idx[0])
        i_idx = torch.unique(idx[1])
        user_cl_loss = self.InfoNCE(user_view1[u_idx], user_view2[u_idx], self.args.temp)  # 只有一个超参数temp
        item_cl_loss = self.InfoNCE(item_view1[i_idx], item_view2[i_idx], self.args.temp)
        return user_cl_loss + item_cl_loss

    def InfoNCE(self, view1, view2, temperature, b_cos=True):
        if b_cos:
            view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score + 10e-6)
        return torch.mean(cl_loss)

    def Aspect_condition_encoder(self, emb, aspect_emb):
        # new_emb = self.MLP(emb + aspect_emb)
        # new_emb = self.MLP(torch.cat((emb, aspect_emb), dim=1))
        # new_emb = torch.cat((emb, aspect_emb), dim = 1)
        new_emb = emb + aspect_emb
        return new_emb # F.sigmoid(new_emb)  # new_emb # F.sigmoid(new_emb) relu(new_emb) 换成Relu

    # 与未交互过的aspect计算相似度，选出top-k个表示加入到user中
    def aspect_diversity(self, user_emb, user_index):  # user_index: [batch_size]
        # user_emb =ls all_user_emb[user_index]
        # 计算相似度
        aspect_emb = torch.tensor(list(self.aspect_emb_64.values())).to(self.args.device)  # 所有aspect嵌入
        # output: [num_user, num_aspect]
        # 换成余弦相似度 或者 点乘  或者 加权相加
        # dicts = torch.cdist(user_emb, aspect_emb) # user_emb: [batch_user, dim]  [num_aspect, dim] result： [batch_user, num_aspect]
        dicts = self.cos_sim(user_emb, aspect_emb)
        mask = torch.index_select(self.aspect_mask, 0, user_index)  # 选出user对应的aspect交互列表 [batch_size, n_aspect]
        masked_dicts = mask + dicts  # 将已交互过的aspect的分数设置为负无穷
        _, sorted_index = torch.sort(masked_dicts, dim=1, descending=True)  # 得到排序后的Aaspect的编码
        # 加入new_aspects个新的aspect
        sorted_index = sorted_index[:, : self.args.new_aspects]
        new_user_emb = []
        # 批量选取新的aspects
        new_aspect_emb = []
        for i in range(len(sorted_index)):
            new_aspect_emb.append(torch.index_select(aspect_emb, 0, sorted_index[i]))  # 每个user对应的new aspect

        new_aspect_emb = torch.stack(new_aspect_emb)  # torch.Size([2048, 8, 64])
        new_aspect_emb = torch.mean(new_aspect_emb, 1)
        # new_user_emb.append((user_emb[user_index] + new_aspect_emb) / 2)  # 取平均
        # 这里改成加权
        new_user_emb = ((user_emb + new_aspect_emb) / 2).to(self.args.device)
        # new_user_emb = (user_emb + self.new_λ * new_aspect_emb).to(self.args.device)

        # new_user_emb = torch.stack(new_user_emb).to(self.args.device)
        return new_user_emb

    def aspect_diversity2(self, user_emb, user_index):
        # self.new_aspect_emb: torch.Size([num_user, num_new_aspects, emb])
        new_aspect_emb = torch.index_select(self.new_aspect_emb, 0,
                                            user_index)  # torch.Size([user_index, num_new_aspects, emb])
        # [n_user, num_new_aspects, 1]
        new_aspect_weight = nn.Softmax(dim=1)(
            torch.matmul(torch.unsqueeze(user_emb, dim=1), new_aspect_emb.transpose(1, 2))).squeeze().unsqueeze(dim=-1)
        user_new_aspect = self.new_λ * user_emb * (new_aspect_weight * new_aspect_emb).sum(dim=1) + user_emb
        # user_new_aspect = self.new_λ  * (new_aspect_weight * new_aspect_emb).sum(dim=1) + user_emb
        return user_new_aspect

    # 计算两个二维向量每行之间的余弦相似度
    def cos_sim(self, emb1, emb2):
        new_emb1 = F.normalize(emb1, dim=1)
        new_emb2 = F.normalize(emb2, dim=1)
        cos = torch.mm(new_emb1, new_emb2.transpose(0, 1))
        return cos


# submodular的时候考虑上aspect
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

    # submodular  加入了aspect的嵌入
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