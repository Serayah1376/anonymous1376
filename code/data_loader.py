import os
from os.path import join
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix
import scipy.sparse as sp
from time import time
from gensim.models import Word2Vec
import json
import code.utils as utils
import parser
import dgl


class BasicDataset(Dataset):
    def __init__(self):
        print("init dataset")

    @property
    def n_users(self):
        raise NotImplementedError

    @property
    def m_items(self):
        raise NotImplementedError

    @property
    def trainDataSize(self):
        raise NotImplementedError

    @property
    def testDict(self):
        raise NotImplementedError

    @property
    def allPos(self):
        raise NotImplementedError

    def read_category(self):
        raise NotImplementedError

    def getAspect(self):
        raise NotImplementedError

    def getUserItemFeedback(self, users, items):
        raise NotImplementedError

    def getUserPosItems(self, users):
        raise NotImplementedError

    def getUserNegItems(self, users):
        """
        not necessary for large dataset
        it's stupid to return all neg items in super large dataset
        """
        raise NotImplementedError

    def getSparseGraph(self):
        """
        build a graph in torch.sparse.IntTensor.
        Details in NGCF's matrix form
        A =
            |I,   R|
            |R^T, I|
        """
        raise NotImplementedError


# 加载数据 dataset
class Loader(BasicDataset):

    def __init__(self, args, path="../data/yelp2018"):
        # train or test
        print(f'loading [{path}]')
        self.args = args
        self.split = args.A_split
        self.folds = args.a_fold
        self.mode_dict = {'train': 0, "test": 1}
        self.mode = self.mode_dict['train']
        self.n_user = 0  # user数量
        self.m_item = 0  # item数量
        train_file = path + '/train.txt'
        test_file = path + '/test.txt'

        user_aspect = path + '/user_aspect.json'
        item_aspect = path + '/item_aspect.json'

        aspect_embedding = path + '/aspect_all-MiniLM-L6-v2.json'

        self.category_path = path + '/bussiness_category.json'

        self.path = path
        trainUniqueUsers, trainItem, trainUser = [], [], []
        testUniqueUsers, testItem, testUser = [], [], []
        self.user_aspect_dic = dict()
        self.item_aspect_dic = dict()
        self.aspect_emb = dict()

        self.traindataSize = 0
        self.testDataSize = 0

        self.category_dic, self.category_num = self.read_category(self.category_path)

        # 训练数据
        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    trainUniqueUsers.append(uid)
                    trainUser.extend([uid] * len(items))  # 与交互的item对应
                    trainItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.traindataSize += len(items)
        self.trainUniqueUsers = np.array(trainUniqueUsers)  # 所有的userID
        self.trainUser = np.array(trainUser)
        self.trainItem = np.array(trainItem)

        # 测试数据
        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    testUniqueUsers.append(uid)
                    testUser.extend([uid] * len(items))
                    testItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.testDataSize += len(items)
        self.m_item += 1  # 因为ID编码是从0开始的
        self.n_user += 1
        self.testUniqueUsers = np.array(testUniqueUsers)
        self.testUser = np.array(testUser)
        self.testItem = np.array(testItem)

        # 所有的user、item ID 去重
        self.all_user = np.unique(np.append(self.trainUniqueUsers, self.testUniqueUsers))
        self.all_item = np.unique(np.append(self.trainItem, self.testItem))

        # 使用模型all-MiniLM-L6-v2处理的aspect嵌入
        with open(aspect_embedding) as f:
            for l in f.readlines():
                dic = json.loads(l)
                self.aspect_emb[dic['aspect']] = dic['embedding']

        # user和aspect字符列表的对应
        with open(user_aspect) as f:
            for l in f.readlines():
                dic = json.loads(l)
                self.user_aspect_dic[int(dic["userID"])] = dic["aspects"]  # user和aspect对应列表 asepct: list

        # item和aspect字符列表的对应
        with open(item_aspect) as f:
            for l in f.readlines():
                dic = json.loads(l)
                self.item_aspect_dic[int(dic["itemID"])] = dic["aspects"]  # item和aspect对应列表

        # user/item对应的aspect的嵌入 key: user   value: [n_aspect, aspect_emb]
        self.user_aspect_embedding = dict()
        self.item_aspect_embedding = dict()

        self.Graph = None
        print(f"{self.trainDataSize} interactions for training")
        print(f"{self.testDataSize} interactions for testing")
        print(f"{args.dataset} Sparsity : {(self.trainDataSize + self.testDataSize) / self.n_users / self.m_items}")

        # (users,items), bipartite graph， 交互稀疏矩阵
        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                      shape=(self.n_user, self.m_item))
        # ?
        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.
        # pre-calculate
        self._allPos = self.getUserPosItems(list(range(self.n_user)))  # 每个user交互过的item
        self.__testDict = self.__build_test()
        print(f"{args.dataset} is ready to go")

    @property
    def n_users(self):
        return self.n_user

    @property
    def m_items(self):
        return self.m_item

    @property
    def trainDataSize(self):
        return self.traindataSize

    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos

    def _split_A_hat(self, A):
        A_fold = []
        fold_len = (self.n_users + self.m_items) // self.folds
        for i_fold in range(self.folds):
            start = i_fold * fold_len
            if i_fold == self.folds - 1:
                end = self.n_users + self.m_items
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(self.args.device))
        return A_fold

    # 得到item与类别的对应
    def read_category(self, path):
        dic = {}
        all_category = []
        f = open(path, 'r').readlines()
        for l in f:
            tmp = json.loads(l)
            item = tmp['business_remap_id']
            category = tmp['categoriesID']
            all_category.extend(category)
            dic[item] = category
        all_category = list(set(all_category))
        num = len(all_category)
        return dic, num

        # 获得稀疏交互矩阵

    def getSparseGraph(self):
        print("loading adjacency matrix")
        if self.Graph is None:  # True
            try:  # 无
                pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat.npz')
                print("successfully loaded...")
                norm_adj = pre_adj_mat
            except:
                print("generating adjacency matrix")
                s = time()
                # n_nodes, 双向交互矩阵
                adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
                adj_mat = adj_mat.tolil()  # 转换成列表
                R = self.UserItemNet.tolil()
                adj_mat[:self.n_users, self.n_users:] = R
                adj_mat[self.n_users:, :self.n_users] = R.T
                adj_mat = adj_mat.todok()
                edge_src, edge_dst = adj_mat.nonzero()  # 双向的

                self.Graph = dgl.graph(data=(edge_src, edge_dst),
                                       idtype=torch.int32,
                                       num_nodes=adj_mat.shape[0],
                                       device=self.args.device)

        return self.Graph

    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        # print(self.UserItemNet[users, items])
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))

    def getUserPosItems(self, users):  # user list
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])  # 不为0的col和raw
        return posItems

