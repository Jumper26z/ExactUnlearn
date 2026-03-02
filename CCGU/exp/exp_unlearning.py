import logging
import time
import numpy as np

from exp.exp import Exp
from lib_gnn_model.NGCF.NGCF_run import NGCF_run
from lib_gnn_model.SGL.SGL_run import SGL_run
from lib_gnn_model.lightgcn.LightGCN_run import LightGCN_run
from lib_aggregator.aggregator import Aggregator

class ExpUnlearning(Exp):
    def __init__(self, args):
        super(ExpUnlearning, self).__init__(args)

        self.logger = logging.getLogger('exp_unlearning')

        self.target_model_name = self.args['target_model']
        self.num_opt_samples = self.args['num_opt_samples']

        self.load_data()
        self.determine_target_model()

        all_scores = []
        unlearning_time = np.empty((0))
        for run in range(self.args['num_runs']):  # num_runs:1
            self.logger.info("Run %f" % run)
            self.train_target_models(run)
            aggregate_score = self.aggregate(run)
            all_scores.append(aggregate_score)
            node_unlearning_time = self.unlearning_time_statistic()
            unlearning_time = np.append(unlearning_time, node_unlearning_time)

        all_scores = np.array(all_scores)
        # 计算均值和标准差
        precision_avg, recall_avg, mrr_avg, ndcg_avg = np.mean(all_scores, axis=0)
        precision_std, recall_std, mrr_std, ndcg_std = np.std(all_scores, axis=0)
        self.score_avg = [precision_avg, recall_avg, mrr_avg, ndcg_avg]
        self.score_std = [precision_std, recall_std, mrr_std, ndcg_std]
        self.unlearning_time_avg = np.average(unlearning_time)
        self.unlearning_time_std = np.std(unlearning_time)
        self.logger.info("Avg: %s, Std: %s, Time Avg: %.4f, Time Std: %.4f" % (self.score_avg, self.score_std, self.unlearning_time_avg, self.unlearning_time_std))

    def load_data(self):
        self.shard_data = self.data_store.load_shard_data()
        self.data = self.data_store.load_raw_data()

    def determine_target_model(self):
        # Based on the 'target_model' argument, we determine which model to use.
        if self.target_model_name == 'LightGCN':
            self.target_model = LightGCN_run
        elif self.target_model_name == 'SGL':
            self.target_model = SGL_run
        elif self.target_model_name == 'NGCF':
            self.target_model = NGCF_run
        else:
            raise Exception(f"Unsupported target model: {self.target_model_name}")

    def train_target_models(self, run):
        if self.args['is_train_target_model']:
            self.logger.info('training target models')

            self.time = {}
            for shard in range(self.args['num_shards']):
                self.time[shard] = self._train_model(run, shard)

    def aggregate(self, run):
        self.logger.info('aggregating submodels')

        start_time = time.time()
        aggregator = Aggregator(run, self.target_model, self.data, self.shard_data, self.args)
        aggregator.generate_posterior()
        self.precision, self.recall, self.mrr, self.ndcg = aggregator.aggregate()
        aggregate_time = time.time() - start_time
        self.logger.info("Partition cost %s seconds." % aggregate_time)
        k = self.args['top_k']
        self.logger.info(f"Metrics@{k}: Precision: {self.precision:.4f}, Recall: {self.recall:.4f}, MRR: {self.mrr:.4f}, NDCG: {self.ndcg:.4f}")

        return self.precision, self.recall, self.mrr, self.ndcg

    def unlearning_time_statistic(self):
        if self.args['is_train_target_model'] and self.args['num_shards'] != 1:
            self.community_to_node = self.data_store.load_community_data()
            node_list = []
            for key, value in self.community_to_node.items():
                node_list.extend(value)

            # random sample 5% nodes, find their belonging communities
            sample_nodes = np.random.choice(node_list, int(0.05 * len(node_list)))
            belong_community = []
            for sample_node in range(len(sample_nodes)):
                for community, node in self.community_to_node.items():
                    if np.in1d(sample_nodes[sample_node], node).any():
                        belong_community.append(community)

            # calculate the total unlearning time and group unlearning time
            group_unlearning_time = []
            node_unlearning_time = []
            for shard in range(self.args['num_shards']):
                if belong_community.count(shard) != 0:
                    group_unlearning_time.append(self.time[shard])
                    node_unlearning_time.extend([float(self.time[shard]) for j in range(belong_community.count(shard))])

            return node_unlearning_time

        elif self.args['is_train_target_model'] and self.args['num_shards'] == 1:
            return self.time[0]

        else:
            return 0

    def _train_model(self, run, shard):
        self.logger.info('training target models, run %s, shard %s' % (run, shard))
        start_time = time.time()

        # 取出当前 shard 的数据
        shard_data = self.shard_data[shard]

        # 根据选择的模型类型来实例化不同的模型
        if self.target_model_name == 'LightGCN':
            model = LightGCN_run(
                num_users=shard_data.num_users,
                num_items=shard_data.num_items,
                hidden_channels=self.args['hidden_channels'],
                num_layers=self.args['num_layers'],
            )
        elif self.target_model_name == 'SGL':
            model = SGL_run(
                num_users=shard_data.num_users,
                num_items=shard_data.num_items,
                hidden_channels=self.args['hidden_channels'],
                num_layers=self.args['num_layers'],
            )
        elif self.target_model_name == 'NGCF':
            model = NGCF_run(
                num_users=shard_data.num_users,
                num_items=shard_data.num_items,
                hidden_channels=self.args['hidden_channels'],
                num_layers=self.args['num_layers'],
            )
        else:
            raise Exception(f"Unsupported target model: {self.target_model_name}")

        model.data = shard_data

        model.train_model()

        self.data_store.save_target_model(run, model, shard)
        train_time = time.time() - start_time
        self.logger.info(f"Model training time for shard {shard}: {train_time:.2f}s")
        return train_time