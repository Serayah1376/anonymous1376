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
    early_stop = utils.early_stop(args)

    utils.set_seed(args.seed)
    print(">>SEED:", args.seed)

    dataset = dataloader.Loader(args, dataname=args.dataset, path="./datasets/" + args.dataset)

    # Initialize the model
    Recmodel = utils.choose_model(args, dataset)
    Recmodel = Recmodel.to(args.device)
    early_stop(999999.99, Recmodel)
    Recmodel.aspect_init()  # initialization pre_trained aspect

    bpr = utils.BPRLoss(args, Recmodel)

    # 现在先不管，等模型确定之后在弄
    weight_file = utils.getFileName(args)  # 权重
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
        for epoch in range(args.epochs + 1):
            if epoch % 100 == 0 and epoch != 0:
                test_start = time.time()
                print("[TEST]")
                tester = Tester(args, dataset, Recmodel, epoch, w, args.multicore)
                tester.test()
                test_end = time.time()
                print("test time:", test_end - test_start)

            train_start = time.time()
            output_information, loss = trainer.train(args, dataset, Recmodel, bpr, epoch, neg_k=Neg_k, w=w)  # main
            train_end = time.time()
            print(f'EPOCH[{epoch + 1}/{args.epochs}] {output_information}use_time:{train_end - train_start}')
            # torch.save(Recmodel.state_dict(), weight_file) # 模型确定之后再改
            early_stop(loss, Recmodel)
            if early_stop.early_stop:
                break
    finally:
        if args.tensorboard:
            w.close()
