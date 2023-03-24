import numpy as np
import torch
import code
import code.utils as utils
import code.model as model
import multiprocessing
from .data_loader import BasicDataset
from sklearn.metrics import roc_auc_score

CORES = multiprocessing.cpu_count() // 2


class Tester(object):
    def __init__(self, args, dataset, Recmodel, epoch, w=None, multicore=0):
        self.args = args
        self.u_batch_size = args.testbatch
        self.dataset: BasicDataset = dataset
        self.testDict: dict = dataset.testDict
        Recmodel: model.LightGCN
        # eval mode with no dropout
        self.Recmodel = Recmodel.eval()
        self.topks = eval(args.topks)
        self.max_K = max(self.topks)
        self.multicore = multicore
        self.w = w
        self.epoch = epoch
        if self.multicore == 1:
            self.pool = multiprocessing.Pool(CORES)
        self.results = {'precision': np.zeros(len(self.topks)),
                        'recall': np.zeros(len(self.topks)),
                        'ndcg': np.zeros(len(self.topks)),
                        'coverage': np.zeros(len(self.topks)),
                        'ILD': np.zeros(len(self.topks))}  # 待确定

        self.cate = np.array(list(dataset.category_dic.values()))  # 对于yelp2018来说是二维数组，因为每个item有多种类别

    def test_one_batch(self, X):
        sorted_items = X[0].numpy()
        groundTrue = X[1]
        r = self.getLabel(groundTrue, sorted_items)
        stat = self.stat(sorted_items)
        pre, recall, ndcg, coverage, ILD = [], [], [], [], []
        for k in self.topks:
            ret = self.RecallPrecision_ATk(groundTrue, r, k)
            pre.append(ret['precision'])
            recall.append(ret['recall'])
            ndcg.append(self.NDCGatK_r(groundTrue, r, k))
            coverage.append(self.coverage(stat,k))
            ILD.append(self.ILD(stat, r, k))
        return {'recall': np.array(recall),
                'precision': np.array(pre),
                'ndcg': np.array(ndcg),
                'caverage': np.array(coverage),
                'ILD': np.array(self.ILD)}

    def test(self):
        with torch.no_grad():
            users = list(self.testDict.keys())  # 测试的user列表
            try:
                assert self.u_batch_size <= len(users) / 10
            except AssertionError:
                print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
            users_list = []
            rating_list = []
            groundTrue_list = []
            # auc_record = []
            # ratings = []
            total_batch = len(users) // self.u_batch_size + 1
            for batch_users in utils.minibatch(users, batch_size=self.u_batch_size):
                allPos = self.dataset.getUserPosItems(batch_users)  # 交互items
                all_items = self.dataset.all_item
                groundTrue = [self.testDict[u] for u in batch_users]
                batch_users_gpu = torch.Tensor(batch_users).long()
                batch_users_gpu = batch_users_gpu.to(self.args.device)

                rating = self.Recmodel.getUsersRating(batch_users_gpu, all_items)
                # rating = rating.cpu()
                exclude_index = []
                exclude_items = []
                for range_i, items in enumerate(allPos):
                    exclude_index.extend([range_i] * len(items))
                    exclude_items.extend(items)
                rating[exclude_index, exclude_items] = -(1 << 10)
                _, rating_K = torch.topk(rating, k=self.max_K)  # 选最大的k
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
            X = zip(rating_list, groundTrue_list)  # 预测和真值
            if self.multicore == 1:
                pre_results = self.pool.map(self.test_one_batch, X)
            else:
                pre_results = []
                for x in X:
                    pre_results.append(self.test_one_batch(x))
            scale = float(self.u_batch_size / len(users))
            for result in pre_results:
                self.results['recall'] += result['recall']
                self.results['precision'] += result['precision']
                self.results['ndcg'] += result['ndcg']
                self.results['coverage'] += result['coverage']
            self.results['recall'] /= float(len(users))
            self.results['precision'] /= float(len(users))
            self.results['ndcg'] /= float(len(users))
            self.results['coverage'] /= float(len(users))
            # results['auc'] = np.mean(auc_record)
            if self.args.tensorboard:
                self.w.add_scalars(f'Test/Recall@{self.topks}',
                              {str(self.topks[i]): self.results['recall'][i] for i in range(len(self.topks))}, self.epoch)
                self.w.add_scalars(f'Test/Precision@{self.topks}',
                              {str(self.topks[i]): self.results['precision'][i] for i in range(len(self.topks))}, self.epoch)
                self.w.add_scalars(f'Test/NDCG@{self.topks}',
                              {str(self.topks[i]): self.results['ndcg'][i] for i in range(len(self.topks))}, self.epoch)
            if self.multicore == 1:
                self.pool.close()
            print(self.results)
            return self.results

    def getLabel(self, test_data, pred_data):
        r = []
        for i in range(len(test_data)):
            groundTrue = test_data[i]
            predictTopK = pred_data[i]
            pred = list(map(lambda x: x in groundTrue, predictTopK))  # examples: [True, True, False...]
            pred = np.array(pred).astype("float")
            r.append(pred)
        return np.array(r).astype('float')

    # ====================Metrics=============================
    # Accuracy
    def RecallPrecision_ATk(test_data, r, k):
        """
        test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
        pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
        k : top-k
        """
        right_pred = r[:, :k].sum(1)
        precis_n = k
        recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
        recall = np.sum(right_pred / recall_n)
        precis = np.sum(right_pred) / precis_n
        return {'recall': recall, 'precision': precis}

    def MRRatK_r(r, k):
        """
        Mean Reciprocal Rank
        """
        pred_data = r[:, :k]
        scores = np.log2(1. / np.arange(1, k + 1))
        pred_data = pred_data / scores
        pred_data = pred_data.sum(1)
        return np.sum(pred_data)

    def NDCGatK_r(self, test_data, r, k):
        """
        Normalized Discounted Cumulative Gain
        rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
        """
        assert len(r) == len(test_data)
        pred_data = r[:, :k]

        test_matrix = np.zeros((len(pred_data), k))
        for i, items in enumerate(test_data):
            length = k if k <= len(items) else len(items)
            test_matrix[i, :length] = 1
        max_r = test_matrix
        idcg = np.sum(max_r * 1. / np.log2(np.arange(2, k + 2)), axis=1)
        dcg = pred_data * (1. / np.log2(np.arange(2, k + 2)))
        dcg = np.sum(dcg, axis=1)
        idcg[idcg == 0.] = 1.
        ndcg = dcg / idcg
        ndcg[np.isnan(ndcg)] = 0.
        return np.sum(ndcg)

    def AUC(self, all_item_scores, dataset, test_data):
        """
            design for a single user
        """
        dataset: BasicDataset
        r_all = np.zeros((dataset.m_items,))
        r_all[test_data] = 1
        r = r_all[all_item_scores >= 0]
        test_item_scores = all_item_scores[all_item_scores >= 0]
        return roc_auc_score(r, test_item_scores)

    # diversity
    # top-k 商品中的种类数量
    def coverageatK_r(self, stat, k):
        cate_list = []
        for i in range(k):
            cate_list.extend(stat[i])  # top-k个商品的种类数
        num = np.unique(np.array(cate_list)).size
        return num

    def stat(self, items):
        stat = [list(np.unique(self.cate[item], return_counts=True)[0]) for item in items]
        return stat

    # 需要检查
    def ILD(ranklist, item_cate_dict):
        K = len(ranklist)
        item_cate_arr = np.array(list(item_cate_dict.values()))
        tmp = 0
        catelist = item_cate_arr[ranklist]
        for i in range(len(catelist)):
            for j in range(i, len(catelist)):
                if catelist[i] != catelist[j]:
                    tmp += 1  # 两个item的category vec计算cos距离，相当于相同category的加上1
        tmp = 2 * tmp / (K * (K - 1))
        return tmp



