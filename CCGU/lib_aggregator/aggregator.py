import logging
import torch
from tensorflow.python.ops.nn_ops import top_k

from lib_gnn_model.lightgcn.LightGCN_run import LightGCN_run
from lib_gnn_model.SGL.SGL_run import SGL
from lib_gnn_model.NGCF.NGCF_run import NGCF

torch.cuda.empty_cache()

from sklearn.metrics import f1_score
import numpy as np

from lib_aggregator.optimal_aggregator import OptimalAggregator
from lib_dataset.data_store import DataStore
import math


class Aggregator:
    def __init__(self, run, target_model_name, data, shard_data, args):
        self.global_posterior = None
        self.logger = logging.getLogger('Aggregator')
        self.args = args

        self.data_store = DataStore(self.args)

        self.run = run
        self.target_model_name = target_model_name
        self.data = data
        self.shard_data = shard_data

        self.num_shards = args['num_shards']

        self.determine_target_model()

    def determine_target_model(self):
        # Dynamically select the model based on the target_model_name argument
        if self.target_model_name == 'LightGCN':
            self.target_model_class = LightGCN_run
        elif self.target_model_name == 'SGL':
            self.target_model_class = SGL
        elif self.target_model_name == 'NGCF':
            self.target_model_class = NGCF
        else:
            raise Exception(f"Unsupported target model: {self.target_model_name}")

    def generate_posterior(self, suffix=""):
        self.true_label = self.data.edge_label_index.detach().cpu().numpy()
        self.label_mat = torch.zeros((self.data.num_users, self.data.num_items), dtype=torch.bool)
        users = self.true_label[0]
        items = self.true_label[1]
        self.label_mat[users - self.data.num_items, items] = True

        self.posteriors = {}
        mappings = self.data_store.load_shard_mappings()

        num_users = self.data.num_users
        num_items = self.data.num_items

        self.global_posterior = torch.full((num_users, num_items), 0)

        for shard in range(self.args['num_shards']):
            mapping = mappings[shard]  # new_to_old 的映射
            self.target_model = self.target_model_class(
                num_users=self.shard_data[shard].num_users,
                num_items=self.shard_data[shard].num_items,
                hidden_channels=self.args['hidden_channels'],
                num_layers=self.args['num_layers'],
            )
            self.target_model.data = self.shard_data[shard]
            self.target_model.prepare_for_inference()
            self.data_store.load_target_model(self.run, self.target_model, shard, suffix)
            self.posteriors[shard] = self.target_model.posterior()

            shard_num_users = self.shard_data[shard].num_users
            shard_num_items = self.shard_data[shard].num_items

            global_items = [mapping[i] for i in range(shard_num_items)]
            global_users = [mapping[shard_num_items + u] for u in range(shard_num_users)]
            global_users = [gu - self.data.num_items for gu in global_users]  # 转成行索引
            for lu, gu in enumerate(global_users):
                for li, gi in enumerate(global_items):
                    self.global_posterior[gu, gi - num_users] = self.posteriors[shard][lu, li]

        self.logger.info("Saving posteriors.")
        self.data_store.save_posteriors(self.posteriors, self.run, suffix)

    def aggregate(self):
        if self.args['aggregator'] == 'mean':
            precision, recall, mrr, ndcg = self._mean_aggregator()
        elif self.args['aggregator'] == 'optimal':
            aggregate_f1_score = self._optimal_aggregator()
        elif self.args['aggregator'] == 'majority':
            aggregate_f1_score = self._majority_aggregator()
        else:
            raise Exception("unsupported aggregator.")

        return precision, recall, mrr, ndcg

    def _mean_aggregator(self):

        num_users, num_items = self.global_posterior.shape

        precision_sum = 0.0
        recall_sum = 0.0
        mrr_sum = 0.0
        ndcg_sum = 0.0
        valid_users = 0

        for u in range(num_users):
            gt_items = self.label_mat[u].nonzero(as_tuple=True)[0]
            if len(gt_items) == 0:
                continue

            scores = self.global_posterior[u]

            topk_items = torch.topk(scores, self.args['top_k']).indices

            hits = self.label_mat[u][topk_items]

            precision_sum += hits.sum().item() / self.args['top_k']
            recall_sum += hits.sum().item() / len(gt_items)
            if hits.any():
                first_hit_rank = hits.nonzero(as_tuple=True)[0][0].item() + 1  # rank 从 1 开始
                mrr_sum += 1.0 / first_hit_rank
            else:
                mrr_sum += 0.0

            ideal_dcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gt_items), self.args['top_k'])))
            dcg = sum((1.0 / math.log2(i + 2)) if hits[i] else 0.0 for i in range(len(hits)))
            ndcg_sum += dcg / ideal_dcg if ideal_dcg > 0 else 0.0

            valid_users += 1

        precision = precision_sum / valid_users
        recall = recall_sum / valid_users
        mrr = mrr_sum / valid_users
        ndcg = ndcg_sum / valid_users

        return precision, recall, mrr, ndcg

    def _majority_aggregator(self):
        pred_labels = []
        for shard in range(self.num_shards):
            pred_labels.append(self.posteriors[shard].argmax(axis=1).cpu().numpy())

        pred_labels = np.stack(pred_labels)
        pred_label = np.argmax(
            np.apply_along_axis(np.bincount, axis=0, arr=pred_labels, minlength=self.posteriors[0].shape[1]), axis=0)

        return f1_score(self.true_label, pred_label, average="micro")

    def _optimal_aggregator(self):
        optimal = OptimalAggregator(self.run, self.target_model_name, self.data, self.args)
        optimal.generate_train_data()
        weight_para = optimal.optimization()
        self.data_store.save_optimal_weight(weight_para, run=self.run)

        posterior = self.posteriors[0] * weight_para[0]
        for shard in range(1, self.num_shards):
            posterior += self.posteriors[shard] * weight_para[shard]

        return f1_score(self.true_label, posterior.argmax(axis=1).cpu().numpy(), average="micro")