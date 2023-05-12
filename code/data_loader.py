from os.path import join
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix
import scipy.sparse as sp
from time import time
from gensim.models import Word2Vec
import json
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


# load data
class Loader(BasicDataset):

    def __init__(self, args, dataname='yelp2018', path="../data/yelp2018"):
        # train or test
        print(f'loading [{path}]')
        self.args = args
        self.split = args.A_split
        self.folds = args.a_fold
        self.mode_dict = {'train': 0, "test": 1}
        self.mode = self.mode_dict['train']
        self.n_user = 0  # Number of users
        self.m_item = 0  # Number of items
        self.n_aspect = 0

        train_file = path + '/train.txt'
        test_file = path + '/test.txt'

        train_aspect = path + '/beauty_interAspectID.json'

        user_aspect = path + '/user_aspect.json'
        item_aspect = path + '/item_aspect.json'

        # aspect_embedding = path + '/aspect_all-MiniLM-L6-v2.json'
        aspect_embedding = path + '/aspect_instructor.json'

        self.category_path = path + '/item_category.json'

        self.path = path
        trainUniqueUsers, trainItem, trainUser = [], [], []
        testUniqueUsers, testItem, testUser = [], [], []
        trainAuser, trainUAspect = [], []
        trainAitem, trainIAspect = [], []
        self.user_aspect_dic = dict()
        self.item_aspect_dic = dict()
        # aspect trainable
        self.user_aspect_ID_dic = dict()
        self.item_aspect_ID_dic = dict()
        self.aspect_emb = dict()  # aspect2emb  (aspect：str)
        self.aspectID2emb = dict()
        self.inter_aspect = dict()

        self.traindataSize = 0
        self.testDataSize = 0

        # category procassing
        if dataname == 'yelp2018':
            self.category_dic, self.category_num = self.read_category_yelp(self.category_path)
        else:
            self.category_dic, self.category_num = self.read_category_beauty(self.category_path)

        # train_data：[user1, user2,...]  [item1, item2, ...]
        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    trainUniqueUsers.append(uid)
                    trainUser.extend([uid] * len(items))  # correspond to the item
                    trainItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.traindataSize += len(items)
        self.trainUniqueUsers = np.array(trainUniqueUsers)  # 所有的userID
        self.trainUser = np.array(trainUser)
        self.trainItem = np.array(trainItem)

        # test data: [user1, user2,...]  [item1, item2, ...]
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

        self.all_user = np.unique(np.append(self.trainUniqueUsers, self.testUniqueUsers))
        self.all_item = np.unique(np.append(self.trainItem, self.testItem))

        with open(user_aspect) as f:
            for l in f.readlines():
                dic = json.loads(l)
                aspects = dic["aspectID"]
                uid = dic["userID"]
                trainAuser.extend([uid] * len(aspects))  # correspond to the aspect
                trainUAspect.extend(aspects)
                self.n_aspect = max(self.n_aspect, max(aspects))
        self.n_aspect += 1
        self.trainAuser = np.array(trainAuser)  # 和aspect对应的userID
        self.trainUAspect = np.array(trainUAspect)  # 和上面user对应的aspectID

        with open(item_aspect) as f:
            for l in f.readlines():
                dic = json.loads(l)
                aspects = dic["aspectID"]
                iid = dic["itemID"]
                trainAitem.extend([iid] * len(aspects))  # correspond to the aspect
                trainIAspect.extend(aspects)
        self.trainAitem = np.array(trainAitem)  # 和aspect对应的itemID
        self.trainIAspect = np.array(trainIAspect)  # 和上面item对应的aspectID

        # use sentence-transformer/all-MiniLM-L6-v2
        with open(aspect_embedding) as f:  # 是按照顺序来的
            for l in f.readlines():
                dic = json.loads(l)
                self.aspect_emb[dic['aspect']] = dic['embedding']
                self.aspectID2emb[dic['aspectID']] = dic['embedding']

        # user-item : aspect list
        with open(train_aspect) as f:
            for l in f.readlines():
                dic = json.loads(l)
                aspect_emb_list = []
                for i in dic["aspectID"]:
                    aspect_emb_list.append(self.aspectID2emb[i])
                aspect_emb_mean = np.mean(np.array(aspect_emb_list), axis=0).tolist()
                self.inter_aspect[dic["user_item_ID"]] = aspect_emb_mean

        # {userID: [aspect_list]}
        with open(user_aspect) as f:
            for l in f.readlines():
                dic = json.loads(l)
                self.user_aspect_dic[int(dic["userID"])] = dic["aspects"]  # userID 与 aspect单词之间的对应关系
                self.user_aspect_ID_dic[int(dic["userID"])] = dic["aspectID"]  # userID 与 aspectID 对应关系

        # {itemID: [aspect_list]}
        with open(item_aspect) as f:
            for l in f.readlines():
                dic = json.loads(l)
                self.item_aspect_dic[int(dic["itemID"])] = dic["aspects"]  # itemID 与 aspect单词 对应关系
                self.item_aspect_ID_dic[int(dic["itemID"])] = dic["aspectID"]  # itemID 与 AspectID 对应关系

        # shape: [n_aspect, aspect_emb]  value: {aspect: embedding}
        self.user_aspect_embedding = dict()
        self.item_aspect_embedding = dict()

        self.Graph = None
        print(f"{self.trainDataSize} interactions for training")
        print(f"{self.testDataSize} interactions for testing")
        print(f"{args.dataset} Sparsity : {(self.trainDataSize + self.testDataSize) / self.n_users / self.m_items}")

        # (users,items), bipartite graph
        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                      shape=(self.n_user, self.m_item))

        # (users,aspects), bipartite graph
        self.UserAspectNet = csr_matrix((np.ones(len(self.trainAuser)), (self.trainAuser, self.trainUAspect)),
                                        shape=(self.n_user, self.n_aspect))

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

    # dic: {item: category}  num: number of category
    def read_category_yelp(self, path):
        dic = {}
        all_category = []
        f = open(path, 'r').readlines()
        for l in f:
            tmp = json.loads(l)
            item = tmp['itemID']
            category = tmp['categoriesID']
            all_category.extend(category)  # list
            dic[item] = category
        all_category = list(set(all_category))
        num = len(all_category)
        return dic, num

    def read_category_beauty(self, path):
        dic = {}
        all_category = []
        f = open(path, 'r').readlines()
        for l in f:
            tmp = json.loads(l)
            item = tmp['itemID']
            category = tmp['categoriesID']
            all_category.append(category)  # [1]
            dic[item] = category
        all_category = list(set(all_category))
        num = len(all_category)
        return dic, num

    def getSparseGraph(self):
        print("loading adjacency matrix")
        if self.Graph is None:
            # heterograph for submodular
            graph_data = {
                ('user', 'rate', 'item'): (torch.tensor(self.trainUser).long(), torch.tensor(self.trainItem).long()),
                ('item', 'rated by', 'user'): (torch.tensor(self.trainItem).long(), torch.tensor(self.trainUser).long())
            }
            self.Graph = dgl.heterograph(graph_data)
            # distinguish between user and item
            category_tensor = torch.tensor(list(self.category_dic.values()), dtype=torch.long).unsqueeze(
                1)  # [category, 1]
            self.Graph.ndata['category'] = {'item': category_tensor, 'user': torch.zeros(self.n_user, 1) - 1}

        return self.Graph.to(self.args.device)

    # for aspect graph
    def getAspectSparseGraph(self):
        print("loading adjacency matritrix")
        if self.Graph is None:
            # 加上aspect -> user  aspect -> item
            graph_data = {
                ('user', 'rate', 'item'): (torch.tensor(self.trainUser).long(), torch.tensor(self.trainItem).long()),
                ('item', 'rated by', 'user'): (
                torch.tensor(self.trainItem).long(), torch.tensor(self.trainUser).long()),
                ('aspect', 'mentioned by au', 'user'): (
                torch.tensor(self.trainUAspect).long(), torch.tensor(self.trainAuser).long()),
                ('aspect', 'mentioned by ai', 'item'): (
                torch.tensor(self.trainIAspect).long(), torch.tensor(self.trainAitem).long())
            }
            self.Graph = dgl.heterograph(graph_data)
            # distinguish between user and item for submodular
            category_tensor = torch.tensor(list(self.category_dic.values()), dtype=torch.long).unsqueeze(
                1)  # [category, 1]
            self.Graph.ndata['category'] = {'item': category_tensor, 'user': torch.zeros(self.n_user, 1) - 1,
                                            "aspect": torch.zeros(self.n_aspect, 1) - 1}

        return self.Graph.to(self.args.device)

    def __build_test(self):
        """
        return: dict: {user: [items]}
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
        users: shape [-1]
        items: shape [-1]
        return: feedback [-1]
        """
        # print(self.UserItemNet[users, items])
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))

    def getUserPosItems(self, users):  # user list
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems

