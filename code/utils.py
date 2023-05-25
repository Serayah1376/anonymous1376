import torch
from torch import optim
import numpy as np
import code.parser as parser
from time import time
from .model import Model
from sklearn.metrics import roc_auc_score
import os
from collections import defaultdict
import time as Time
import torch.nn.functional as F
import pandas as pd
from torch import nn

'''
loss function、negative sampling、utils
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
                 recmodel):
        self.model = recmodel
        self.args = args
        self.lr = args.lr  # learning rate
        self.opt = optim.Adam(recmodel.parameters(), lr=self.lr)

    def alignment(self, x, y):  # 正反馈交互  在这里面添加交互对应的aspect
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        return (x - y).norm(p=2, dim=1).pow(2).mean()

    def uniformity(self, x):
        x = F.normalize(x, dim=-1)  # x = x / ||x||
        return torch.pdist(x, p=2).pow(2).mul(-2).exp().mean().log()  # 矩阵中每行的欧式距离

    # input: the user/item embedding  form the model
    def ali_uni_loss(self, user_e, item_e):  # [batch_size, dim]
        align = self.alignment(user_e, item_e)
        uniform = (self.uniformity(user_e) + self.uniformity(item_e)) / 2
        loss = align + args.gamma * uniform  # gama: [0.2, 0.5, 1, 2, 5, 10]
        return loss

    def stageOne(self, users, pos, neg, aspect_emb):
        # [batch_size, emb_dim]
        users_emb, pos_emb, neg_emb = self.model.bpr_loss(users, pos, neg, aspect_emb)

        au_loss1 = self.ali_uni_loss(users_emb, pos_emb)

        # MLP for dimensional transformation of aspect and attention
        reg_loss = torch.tensor(0.).to(args.device)
        for name, para in self.model.named_parameters():
            reg_loss += para.pow(2).sum()

        reg_loss = reg_loss * self.args.regloss_decay  # + reg_loss1 * self.args.regloss1_decay # regularization weight
        loss = au_loss1 + reg_loss

        self.opt.zero_grad()
        loss.backward()  # retain_graph=True
        self.opt.step()
        return loss.cpu().item(), reg_loss, au_loss1


# negative sampling
def UniformSample_original(dataset, neg_ratio=1):
    # dataset : BasicDataset
    allPos = dataset.allPos
    start = time()
    if sample_ext:  # pos : neg = 1 : 1
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
    # dataset : BasicDataset
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
        return Model(args, dataset)


def set_seed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


# save model
def getFileName(args):
    # path = f"model_{args.model}_dataset_{args.dataset}_embed_size_{args.embed_size}_regloss_weight_decay_{args.regloss2_decay}_layers_{args.layer}_sigma_{args.sigma}_gamma_{args.gamma}_beta_class_{args.beta_class}"
    if args.model == 'mf':
        file = f"mf-{args.dataset}-{args.recdim}.pth.tar"
    elif args.model == 'lgn':
        file = f"lgn-{args.dataset}-{args.layer}-{args.recdim}.pth.tar"
    return os.path.join(args.path, file)


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


# Padding is applied to the aspect list of a single user/item, the padding is 0, so no mask is needed
def aspect_padding(aspect_emb_list, max_len):
    # no aspect list
    if aspect_emb_list == None:
        res = torch.tensor([0 for i in range(args.recdim)]).repeat(max_len, 1).to(args.device)  # 全部设置为0

    elif len(aspect_emb_list) >= max_len:
        res = aspect_emb_list[:max_len][:]
    elif len(aspect_emb_list) < max_len:
        padding_len = max_len - len(aspect_emb_list)  # paddind length
        padding = torch.tensor([0 for i in range(args.recdim)]).repeat(padding_len, 1).to(args.device)
        res = torch.cat((aspect_emb_list, padding), 0)  # padding
    return res.to(torch.float32)  # after padding


def register(args):
    print('===========config================')
    print("dataset:", args.dataset)
    print("layer num:", args.layer)
    print("recdim:", args.recdim)
    print("model:", args.model)
    print("testbatch", args.testbatch)
    print("topks", args.topks)
    print("epochs", args.epochs)
    print("max_len", args.max_len)
    # print("regloss1_decay", args.regloss1_decay)
    print("regloss2_decay", args.regloss_decay)
    print("head_num", args.head_num)
    print("gamma", args.gamma)
    print("gamma2(submodular)", args.gamma2)
    print("sigma(submodular)", args.sigma)
    print("k(number of aggregated neighbors)", args.k)
    print("num of new aspects", args.new_aspects)
    print("the weight of the new aspects add to user", args.new_λ)
    print("whether add new aspects in training", args.train_add_newAspect)
    print("whether add new aspects in testing", args.test_add_newAspect)
    print("loss function", args.loss_function)
    print('===========end===================')


class EarlyStopping:
    """Early stops the training if train loss doesn't improve after a given patience."""

    def __init__(self, save_path, patience=10, verbose=False, delta=0):
        """
        Args:
            save_path : model save folder
            patience (int): How long to wait after last time validation loss improved.
                            Default: 10
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.train_loss_min = np.Inf
        self.delta = delta

    def __call__(self, train_loss, model):

        score = -train_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(train_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'loss: {train_loss}, EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:  # stop
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(train_loss, model)
            self.counter = 0

    def save_checkpoint(self, train_loss, model):
        '''Saves model when train loss decrease.'''
        if self.verbose:
            print(f'Train loss decreased ({self.train_loss_min:.6f} --> {train_loss:.6f}).  Saving model ...')
        # path = os.path.join(self.save_path, 'best_network.pth')
        torch.save(model.state_dict(), self.save_path)
        self.train_loss_min = train_loss


def early_stop(args):
    path = f"model_{args.model}_dataset_{args.dataset}_embed_size_{args.recdim}_regloss_weight_decay_{args.regloss_decay}_layers_{args.layer}_gmma_{args.gamma}_sigma_{args.sigma}_gamma2_{args.gamma2}.pt.tar"
    if os.path.exists('./logs/' + path + '.log'):
        os.remove('./logs/' + path + '.log')

    early_stop = EarlyStopping(patience=args.patience, save_path=os.path.join(args.path, path))
    return early_stop


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

