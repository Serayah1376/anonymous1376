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

    dataset = dataloader.Loader(args, dataname=args.dataset, path="./datasets/" + args.dataset)

    # Initialize the model
    Recmodel = utils.choose_model(args, dataset)
    Recmodel = Recmodel.to(args.device)
    Recmodel.aspect_init()  # initialization pre_trained aspect

    bpr = utils.BPRLoss(args, Recmodel)

    weight_file = utils.getFileName()  # 权重
    print(f"load and save to {weight_file}")

    if args.load:
        try:
            Recmodel.load_state_dict(torch.load(weight_file, map_location=torch.device('cpu')))
            print(f"loaded model weights from {weight_file}")
        except FileNotFoundError:
            print(f"{weight_file} not exists, start from beginning")
    Neg_k = 1  # Negative sample number

    # init tensorboard
    if args.tensorboard:  # True
        w: SummaryWriter = SummaryWriter(
            join(args.runs_path, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + args.comment))
    else:
        w = None
        print("not enable tensorflowboard")

    utils.register(args)  # print import parameters

    # train & test
    try:
        for epoch in range(args.epochs):
            start = time.time()
            if epoch % 10 == 0:
                print("[TEST]")
                tester = Tester(args, dataset, Recmodel, epoch, w, args.multicore)
                tester.test()
            output_information = trainer.train(args, dataset, Recmodel, bpr, epoch, neg_k=Neg_k, w=w)  # main
            end = time.time()
            print(f'EPOCH[{epoch + 1}/{args.epochs}] {output_information}use_time:{end - start}')
            torch.save(Recmodel.state_dict(), weight_file)
    finally:
        if args.tensorboard:
            w.close()
