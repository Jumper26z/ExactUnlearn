import os
import pickle
import logging
import shutil
import config
from torch_geometric.datasets import Planetoid, Coauthor, AmazonBook, MovieLens1M
import torch_geometric.transforms as T
from torch_geometric.datasets import MovieLens100K
from torch_geometric.transforms import ToUndirected
import torch
from torch_geometric.data import HeteroData
import random


class DataStore:
    def __init__(self, args):
        self.logger = logging.getLogger('data_store')
        self.args = args

        self.dataset_name = self.args['dataset_name']
        self.num_features = {
            "cora": 1433,
            "pubmed": 500,
            "citeseer": 3703,
            "Coauthor_CS": 6805,
            "Coauthor_Phys": 8415
        }
        self.partition_method = self.args['partition_method']
        self.num_shards = self.args['num_shards']
        self.target_model = self.args['target_model']

        self.determine_data_path()

    def determine_data_path(self):
        embedding_name = '_'.join(('embedding', self._extract_embedding_method(self.partition_method),
                                   str(self.args['ratio_deleted_edges'])))

        community_name = '_'.join(('community', self.partition_method, str(self.num_shards),
                                   str(self.args['ratio_deleted_edges'])))
        shard_name = '_'.join(('shard_data', self.partition_method, str(self.num_shards),
                               str(self.args['shard_size_delta']), str(self.args['ratio_deleted_edges'])))
        target_model_name = '_'.join((self.target_model, self.partition_method, str(self.num_shards),
                                      str(self.args['shard_size_delta']), str(self.args['ratio_deleted_edges'])))
        optimal_weight_name = '_'.join((self.target_model, self.partition_method, str(self.num_shards),
                                        str(self.args['shard_size_delta']), str(self.args['ratio_deleted_edges'])))

        processed_data_prefix = config.PROCESSED_DATA_PATH + self.dataset_name + "/"#前缀
        self.train_test_split_file =  processed_data_prefix + "train_test_split" + str(self.args['test_ratio'])
        self.train_data_file = processed_data_prefix + "train_data"
        self.train_graph_file = processed_data_prefix + "train_graph"
        self.embedding_file = processed_data_prefix + embedding_name
        self.community_file = processed_data_prefix + community_name
        self.shard_file = processed_data_prefix + shard_name
        self.mapping_file = processed_data_prefix + "shard_node_mappings.pkl"
        self.unlearned_file = processed_data_prefix+ '_'.join(('unlearned', str(self.args['num_unlearned_nodes'])))

        self.target_model_file = config.MODEL_PATH + self.dataset_name + '/' + target_model_name
        self.optimal_weight_file = config.ANALYSIS_PATH + 'optimal/' + self.dataset_name + '/' + optimal_weight_name
        self.posteriors_file = config.ANALYSIS_PATH + 'posteriors/' + self.dataset_name + '/' + target_model_name

        dir_lists = [s + self.dataset_name for s in [config.PROCESSED_DATA_PATH,
                                                     config.MODEL_PATH,
                                                     config.ANALYSIS_PATH + 'optimal/',
                                                     config.ANALYSIS_PATH + 'posteriors/']]
        for dir in dir_lists:
            self._check_and_create_dirs(dir)

    def _check_and_create_dirs(self, folder):#检查是否存在目录，不存在就创建
        if not os.path.exists(folder):
            try:
                self.logger.info("checking directory %s", folder)
                os.makedirs(folder, exist_ok=True)
                self.logger.info("new directory %s created", folder)
            except OSError as error:
                self.logger.info("deleting old and creating new empty %s", folder)
                shutil.rmtree(folder)
                os.mkdir(folder)
                self.logger.info("new empty directory %s created", folder)
        else:
            self.logger.info("folder %s exists, do not need to create again.", folder)



    def load_raw_data(self):
        self.logger.info('loading raw data')
        self.transform = T.NormalizeFeatures()

        if self.dataset_name == "Movielens100K":
            dataset = MovieLens100K(root="temp_data/raw_data/Movielens100K",transform=ToUndirected(),pre_transform=self.transform)
            dataset = dataset[0]
            data = dataset.to_homogeneous()
            user_type = 1 #用户在后面
            item_type = 0 #电影在前面

            num_users = (data.node_type == user_type).sum().item()
            num_items = (data.node_type == item_type).sum().item()

            data.num_users = num_users
            data.num_items = num_items
            return data

        elif self.dataset_name == "Movielens1M":
            dataset = MovieLens1M(root="temp_data/raw_data/Movielens1M")

            data = dataset[0]
            data = self.split_movielens1m(data, test_ratio=0.2)
            num_users, num_items = data['user'].num_nodes, data['movie'].num_nodes
            data = data.to_homogeneous()

            data.num_users = num_users
            data.num_items = num_items
            return data

        elif self.dataset_name == "AmazonBook":
            dataset = AmazonBook(root="temp_data/raw_data/AmazonBook",transform=ToUndirected(),pre_transform=self.transform)
            dataset = dataset[0]
            data = dataset.to_homogeneous()
            user_type = 0
            item_type = 1

            num_users = (data.node_type == user_type).sum().item()
            num_items = (data.node_type == item_type).sum().item()

            data.num_users = num_users
            data.num_items = num_items
            return data


        else:
            raise Exception('unsupported dataset')






    def split_movielens1m(self,data: HeteroData, test_ratio, seed=42):

        torch.manual_seed(seed)
        random.seed(seed)

        edge_index = data['user', 'rates', 'movie'].edge_index
        edge_label = data['user', 'rates', 'movie'].rating
        num_edges = edge_index.size(1)

        indices = list(range(num_edges))
        random.shuffle(indices)
        test_size = int(num_edges * test_ratio)
        test_indices = indices[:test_size]
        train_indices = indices[test_size:]

        train_edge_index = edge_index[:, train_indices]
        train_edge_label = edge_label[train_indices]

        test_edge_index = edge_index[:, test_indices]
        test_edge_label = edge_label[test_indices]

        data['user', 'rates', 'movie'].edge_index = train_edge_index
        data['user', 'rates', 'movie'].rating = train_edge_label
        data['user', 'rates', 'movie'].edge_label_index = test_edge_index
        data['user', 'rates', 'movie'].edge_label = test_edge_label

        data['movie', 'rated_by', 'user'].edge_index = train_edge_index.flip([0])
        data['movie', 'rated_by', 'user'].rating = train_edge_label
        data['movie', 'rated_by', 'user'].edge_label_index = test_edge_index.flip([0])
        data['movie', 'rated_by', 'user'].edge_label = test_edge_label
        return data

    def save_train_data(self, train_data):
        self.logger.info('saving train data')
        pickle.dump(train_data, open(self.train_data_file, 'wb'))

    def load_train_data(self):
        self.logger.info('loading train data')
        return pickle.load(open(self.train_data_file, 'rb'))

    def save_train_graph(self, train_data):
        self.logger.info('saving train graph')
        pickle.dump(train_data, open(self.train_graph_file, 'wb'))

    def load_train_graph(self):
        self.logger.info('loading train graph')
        return pickle.load(open(self.train_graph_file, 'rb'))

    def save_train_test_split(self, train_indices, test_indices):
        self.logger.info('saving train test split data')
        pickle.dump((train_indices, test_indices), open(self.train_test_split_file, 'wb'))

    def load_train_test_split(self):
        self.logger.info('loading train test split data')
        return pickle.load(open(self.train_test_split_file, 'rb'))

    def save_embeddings(self, embeddings):
        self.logger.info('saving embedding data')
        pickle.dump(embeddings, open(self.embedding_file, 'wb'))

    def load_embeddings(self):
        self.logger.info('loading embedding data')
        return pickle.load(open(self.embedding_file, 'rb'))

    def save_community_data(self, community_to_node, suffix=''):
        self.logger.info('saving community data')
        pickle.dump(community_to_node, open(self.community_file + suffix, 'wb'))

    def load_community_data(self, suffix=''):
        self.logger.info('loading community data from: %s'%(self.community_file + suffix))
        return pickle.load(open(self.community_file + suffix, 'rb'))

    def c2n_to_n2c(self, community_to_node):
        node_list = []
        for i in range(self.num_shards):
            node_list.extend(list(community_to_node.values())[i])
        node_to_community = {}

        for comm, nodes in dict(community_to_node).items():
            for node in nodes:
                # Map node id back to original graph
                # node_to_community[node_list[node]] = comm
                node_to_community[node] = comm

        return node_to_community

    def save_shard_mappings(self, shard_node_mappings):
        self.logger.info('saving shard mappings')
        pickle.dump(shard_node_mappings, open(self.mapping_file,'wb'))

    def load_shard_mappings(self):
        self.logger.info('loading shard mappings')
        return pickle.load(open(self.mapping_file,'rb'))

    def save_shard_data(self, shard_data):
        self.logger.info('saving shard data')
        pickle.dump(shard_data, open(self.shard_file, 'wb'))

    def load_shard_data(self):
        self.logger.info('loading shard data')
        return pickle.load(open(self.shard_file, 'rb'))

    def load_unlearned_data(self, suffix):
        file_path = '_'.join((self.unlearned_file, suffix))
        self.logger.info('loading unlearned data from %s' % file_path)
        return pickle.load(open(file_path, 'rb'))

    def save_unlearned_data(self, data, suffix):
        self.logger.info('saving unlearned data %s' % suffix)
        pickle.dump(data, open('_'.join((self.unlearned_file, suffix)), 'wb'))

    def save_target_model(self, run, model, shard, suffix=''):
        if self.args["exp"] in ["node_edge_unlearning", "attack_unlearning"]:
            model_path = '_'.join((self.target_model_file, str(shard), str(run), str(self.args['num_unlearned_nodes']))) + suffix
            model.save_model(model_path)
        else:
            model.save_model(self.target_model_file + '_' + str(shard) + '_' + str(run))
            # model.save_model(self.target_model_file + '_' + str(shard))

    def load_target_model(self, run, model, shard, suffix=''):
        if self.args["exp"] == "node_edge_unlearning":
            model.load_model(
                '_'.join((self.target_model_file, str(shard), str(run), str(self.args['num_unlearned_nodes']))))
        elif self.args["exp"] == "attack_unlearning":
            model_path = '_'.join((self.target_model_file, str(shard), str(run), str(self.args['num_unlearned_nodes']))) + suffix
            print("loading target model from:" + model_path)
            device = torch.device('cpu')
            model.load_model(model_path)
            model.device=device
        else:

            model.load_model(self.target_model_file + '_' + str(shard) + '_' + str(0))

    def save_optimal_weight(self, weight, run):
        torch.save(weight, self.optimal_weight_file + '_' + str(run))

    def load_optimal_weight(self, run):
        return torch.load(self.optimal_weight_file + '_' + str(run))

    def save_posteriors(self, posteriors, run, suffix=''):
        torch.save(posteriors, self.posteriors_file + '_' + str(run) + suffix)

    def load_posteriors(self, run):
        return torch.load(self.posteriors_file + '_' + str(run))

    def _extract_embedding_method(self, partition_method):
        return partition_method.split('_')[0]
