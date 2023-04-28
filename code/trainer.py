import numpy as np
import torch
import code
import code.utils as utils
import code.model as model
import time


# train
def train(args, dataset, recommend_model, loss_class, epoch, neg_k=1, w=None):
    Recmodel = recommend_model
    Recmodel.train()
    bpr: utils.BPRLoss = loss_class

    with utils.timer(name="Sample"):
        S = utils.UniformSample_original(dataset)
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()

    users = users.to(args.device)
    posItems = posItems.to(args.device)
    negItems = negItems.to(args.device)
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    total_batch = len(users) // args.bpr_batch + 1
    aver_loss = 0.

    # test
    reg_loss1 = 0.
    reg_loss2 = 0.
    for (batch_i,
         (batch_users,
          batch_pos,
          batch_neg)) in enumerate(utils.minibatch(users,
                                                   posItems,
                                                   negItems,
                                                   batch_size=args.bpr_batch)):

        cri, reg_loss1, reg_loss2, au_loss1 = bpr.stageOne(batch_users, batch_pos, batch_neg)
        aver_loss += cri
        if args.tensorboard:
            w.add_scalar(f'BPRLoss/BPR', cri, epoch * int(len(users) / args.bpr_batch) + batch_i)

    # loss
    if epoch % 10 == 0 and epoch != 0:
        print("reg_loss1:", reg_loss1, "reg_loss2:", reg_loss2, "au_loss:", au_loss1)

    aver_loss = aver_loss / total_batch
    time_info = utils.timer.dict()
    utils.timer.zero()
    return f"loss{aver_loss:.3f}-{time_info}"



