import torch
from torch import optim
import numpy as np
import code.parser as parser
from .data_loader import BasicDataset
from time import time
from .model import PairWiseModel, LightGCN
from sklearn.metrics import roc_auc_score
import os

'''
loss函数、负采样函数、辅助函数
'''

args = parser.parse_args()
try:
    from cppimport import imp_from_filepath
    from os.path import join, dirname

    path = join(dirname(__file__), "sources/sampling.cpp")
    sampling = imp_from_filepath(path)
    sampling.seed(args.seed)
    sample_ext = True
except:
    print("Cpp extension not loaded")
    sample_ext = False


class BPRLoss:
    def __init__(self,
                 args,
                 recmodel: PairWiseModel):
        self.model = recmodel
        self.weight_decay = args.decay
        self.lr = args.lr  # 学习率
        self.opt = optim.Adam(recmodel.parameters(), lr=self.lr)  # 优化器

    def stageOne(self, users, pos, neg):
        loss, reg_loss1 = self.model.bpr_loss(users, pos, neg)
        reg_loss2 = sum(p.pow(2).sum() for p in self.model.parameters())  # 模型参数正则化
        reg_loss = reg_loss1 * 1e-4 + reg_loss2 * 1e-9  # 调整数量级，因为reg_loss1和reg_loss2不是一个数量级的  self.weight_decay
        loss = loss + reg_loss

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.cpu().item()


# 负采样
def UniformSample_original(dataset, neg_ratio=1):
    dataset: BasicDataset
    allPos = dataset.allPos  # 得到与user交互的所有item
    start = time()
    if sample_ext:  # 负采样  一个正样本对应一个负样本
        S = sampling.sample_negative(dataset.n_users, dataset.m_items,
                                     dataset.trainDataSize, allPos, neg_ratio)
    else:
        S = UniformSample_original_python(dataset)
    return S


def UniformSample_original_python(dataset):
    """
    the original impliment of BPR Sampling in LightGCN
    :return:
        np.array
    """
    total_start = time()
    dataset: BasicDataset
    user_num = dataset.trainDataSize
    users = np.random.randint(0, dataset.n_users, user_num)
    allPos = dataset.allPos
    S = []
    sample_time1 = 0.
    sample_time2 = 0.
    for i, user in enumerate(users):
        start = time()
        posForUser = allPos[user]
        if len(posForUser) == 0:
            continue
        sample_time2 += time() - start
        posindex = np.random.randint(0, len(posForUser))
        positem = posForUser[posindex]
        while True:
            negitem = np.random.randint(0, dataset.m_items)
            if negitem in posForUser:
                continue
            else:
                break
        S.append([user, positem, negitem])
        end = time()
        sample_time1 += end - start
    total = time() - total_start
    return np.array(S)


# ===================end samplers==========================


# =====================utils====================================
def choose_model(args, dataset):
    if args.model == 'lgn':
        return LightGCN(args, dataset)


def set_seed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


# 获得save和load模型的path
def getFileName():
    if args.model == 'mf':
        file = f"mf-{args.dataset}-{args.recdim}.pth.tar"
    elif args.model == 'lgn':
        file = f"lgn-{args.dataset}-{args.layer}-{args.recdim}.pth.tar"
    return os.path.join(args.path, file)


# 分成多个batch
def minibatch(*tensors, **kwargs):
    batch_size = kwargs.get('batch_size', args.bpr_batch)

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)


def shuffle(*arrays, **kwargs):
    require_indices = kwargs.get('indices', False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result


# 对单个user/item的aspect列表进行padding并计算mask矩阵
def aspect_padding(aspect_emb_list, max_len):
    # 如果没有对应的aspect列表
    if aspect_emb_list == None:
        res = torch.tensor([0 for i in range(args.recdim)]).repeat(max_len, 1)  # 全部设置为0
        mask_ones = torch.zeros(max_len)  # 全部mask掉
    elif len(aspect_emb_list) >= max_len:
        res = aspect_emb_list[:max_len]
        mask_ones = torch.ones(max_len)  # 没有进行padding
    elif len(aspect_emb_list) < max_len:
        padding_len = max_len - len(aspect_emb_list)  # padding的长度
        padding = torch.tensor([0 for i in range(args.recdim)]).repeat(padding_len, 1).to(args.device)
        res = torch.cat((aspect_emb_list, padding), 0)  # padding
        mask_ones = torch.cat((torch.ones(len(aspect_emb_list)), torch.zeros(padding_len)), 0)  # padding的部分全为0
    return res.to(args.device), mask_ones.to(args.device)  # padding之后的每个user的aspect列表，以及padding的mask矩阵


def register(args):
    print('===========config================')
    print("layer num:", args.layer)
    print("recdim:", args.recdim)
    print("model:", args.model)
    print("dataset:", args.dataset)
    print("using bpr loss")
    print('===========end===================')


class timer:
    """
    Time context manager for code block
        with timer():
            do something
        timer.get()
    """
    from time import time
    TAPE = [-1]  # global time record
    NAMED_TAPE = {}

    @staticmethod
    def get():
        if len(timer.TAPE) > 1:
            return timer.TAPE.pop()
        else:
            return -1

    @staticmethod
    def dict(select_keys=None):
        hint = "|"
        if select_keys is None:
            for key, value in timer.NAMED_TAPE.items():
                hint = hint + f"{key}:{value:.2f}|"
        else:
            for key in select_keys:
                value = timer.NAMED_TAPE[key]
                hint = hint + f"{key}:{value:.2f}|"
        return hint

    @staticmethod
    def zero(select_keys=None):
        if select_keys is None:
            for key, value in timer.NAMED_TAPE.items():
                timer.NAMED_TAPE[key] = 0
        else:
            for key in select_keys:
                timer.NAMED_TAPE[key] = 0

    def __init__(self, tape=None, **kwargs):
        if kwargs.get('name'):
            timer.NAMED_TAPE[kwargs['name']] = timer.NAMED_TAPE[
                kwargs['name']] if timer.NAMED_TAPE.get(kwargs['name']) else 0.
            self.named = kwargs['name']
            if kwargs.get("group"):
                # TODO: add group function
                pass
        else:
            self.named = False
            self.tape = tape or timer.TAPE

    def __enter__(self):
        self.start = timer.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.named:
            timer.NAMED_TAPE[self.named] += timer.time() - self.start
        else:
            self.tape.append(timer.time() - self.start)

