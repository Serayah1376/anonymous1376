import code.utils as utils
import code.parser as parser
import torch
from tensorboardX import SummaryWriter
import time
from os.path import join
import code.data_loader as dataloader
import code.trainer as trainer
from code.evaluater import Tester
import copy

if __name__ == '__main__':
    args = parser.parse_args()

    utils.set_seed(args.seed)
    print(">>SEED:", args.seed)

    dataset = dataloader.Loader(args, path="./datasets/" + args.dataset)

    # Initialize the model
    deepcopy_dataset = copy.deepcopy(dataset)  # 对数据集进行深拷贝
    Recmodel = utils.choose_model(args, deepcopy_dataset)  # 使用什么model
    Recmodel = Recmodel.to(args.device)
    Recmodel.aspect_init()  # 初始化aspect的处理

    bpr = utils.BPRLoss(args, Recmodel)

    # 权重
    weight_file = utils.getFileName()
    print(f"load and save to {weight_file}")

    if args.load:  # train的时候不加载weight
        try:
            Recmodel.load_state_dict(torch.load(weight_file, map_location=torch.device('cpu')))
            print(f"loaded model weights from {weight_file}")
        except FileNotFoundError:
            print(f"{weight_file} not exists, start from beginning")
    Neg_k = 1

    # init tensorboard
    if args.tensorboard:  # True
        w: SummaryWriter = SummaryWriter(
            join(args.runs_path, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + args.comment))
    else:
        w = None
        print("not enable tensorflowboard")

    utils.register(args)

    # train & test
    try:
        for epoch in range(args.epochs):
            start = time.time()
            if epoch % 10 == 0:  # epoch != 0:
                print("[TEST]")
                tester = Tester(args, dataset, Recmodel, epoch, w, args.multicore)
                tester.test()
            output_information = trainer.train(args, dataset, Recmodel, bpr, epoch, neg_k=Neg_k, w=w)  # main
            print(f'EPOCH[{epoch + 1}/{args.epochs}] {output_information}')
            torch.save(Recmodel.state_dict(), weight_file)
    finally:  # 无论在任何情况下最后都会执行
        if args.tensorboard:
            w.close()
