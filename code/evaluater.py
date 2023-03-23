import numpy as np
import torch
import code
import code.utils as utils
import code.model as model
import multiprocessing

CORES = multiprocessing.cpu_count() // 2

def test_one_batch(args, X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = utils.getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in eval(args.topks):
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(utils.NDCGatK_r(groundTrue, r, k))
    return {'recall': np.array(recall),
            'precision': np.array(pre),
            'ndcg': np.array(ndcg)}

def Test(args, dataset, Recmodel, epoch, w=None, multicore=0):
    u_batch_size = args.testbatch
    dataset: utils.BasicDataset
    testDict: dict = dataset.testDict
    Recmodel: model.LightGCN
    # eval mode with no dropout
    Recmodel = Recmodel.eval()
    topks = eval(args.topks)
    max_K = max(topks)
    if multicore == 1:
        pool = multiprocessing.Pool(CORES)
    results = {'precision': np.zeros(len(topks)),
            'recall': np.zeros(len(topks)),
            'ndcg': np.zeros(len(topks))}
    with torch.no_grad():
        users = list(testDict.keys())
        try:
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        users_list = []
        rating_list = []
        groundTrue_list = []
        # auc_record = []
        # ratings = []
        total_batch = len(users) // u_batch_size + 1
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users)
            all_items = dataset.all_item
            groundTrue = [testDict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(args.device)

            rating = Recmodel.getUsersRating(batch_users_gpu, all_items)
            # rating = rating.cpu()
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1 << 10)
            _, rating_K = torch.topk(rating, k=max_K)
            rating = rating.cpu().numpy()
            # aucs = [
            #         code.AUC(rating[i],
            #                   dataset,
            #                   test_data) for i, test_data in enumerate(groundTrue)
            #     ]
            # auc_record.extend(aucs)
            del rating
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
        assert total_batch == len(users_list)
        X = zip(rating_list, groundTrue_list)
        if multicore == 1:
            pre_results = pool.map(test_one_batch, X)
        else:
            pre_results = []
            for x in X:
                pre_results.append(test_one_batch(args, x))
        scale = float(u_batch_size / len(users))
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        # results['auc'] = np.mean(auc_record)
        if args.tensorboard:
            w.add_scalars(f'Test/Recall@{topks}',
                        {str(topks[i]): results['recall'][i] for i in range(len(topks))}, epoch)
            w.add_scalars(f'Test/Precision@{topks}',
                        {str(topks[i]): results['precision'][i] for i in range(len(topks))}, epoch)
            w.add_scalars(f'Test/NDCG@{topks}',
                        {str(topks[i]): results['ndcg'][i] for i in range(len(topks))}, epoch)
        if multicore == 1:
            pool.close()
        print(results)
        return results


