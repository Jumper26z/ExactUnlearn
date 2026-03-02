import logging
import time
import torch
import numpy as np
from torch_geometric.data import Data
import networkx as nx
from exp.exp import Exp
from lib_graph_partition.graph_partition import GraphPartition

class ExpGraphPartition(Exp):
    def __init__(self, args):
        super(ExpGraphPartition, self).__init__(args)

        self.logger = logging.getLogger('exp_graph_partition')
        self.load_data()
        self.gen_train_graph()
        self.graph_partition()
        self.generate_shard_data()

    def load_data(self):
        self.data = self.data_store.load_raw_data()#return data

    def gen_train_graph(self):
        # delete ratio of edges and update the train graph
        if self.args['ratio_deleted_edges'] != 0:
            self.logger.debug("Before edge deletion. train data  #.Nodes: %f, #.Edges: %f" % (
                self.data.num_nodes, self.data.num_edges))
            nodes = list(range(self.data.num_nodes))
            self.data.edge_index, remain_indices = self._ratio_delete_edges(self.data.edge_index)#剩余的边索引和地址

            if hasattr(self.data, 'rating'):
                self.data.rating = self.data.rating[remain_indices]
            if hasattr(self.data, 'time'):
                self.data.time = self.data.time[remain_indices]
            if hasattr(self.data, 'edge_type'):
                self.data.edge_type = self.data.edge_type[remain_indices]

        edge_index_train = self.data.edge_index.numpy()

        self.train_graph = nx.Graph()
        self.train_graph.add_nodes_from(nodes)

        # reconstruct a networkx train graph
        for u, v in np.transpose(edge_index_train):
            self.train_graph.add_edge(u, v)

        edge_index_np = self.data.edge_index.numpy()
        edges_set = set()
        for u, v in edge_index_np.T:
            edges_set.add(tuple(sorted((u, v))))
        print("删除边后无向边数量（去重后）:", len(edges_set))

        self.logger.debug("After edge deletion. train graph  #.Nodes: %f, #.Edges: %f" % (
            self.train_graph.number_of_nodes(), self.train_graph.number_of_edges()))
        self.logger.debug("After edge deletion. train data  #.Nodes: %f, #.Edges: %f" % (
            self.data.num_nodes, self.data.num_edges))
        self.data_store.save_train_data(self.data)
        self.data_store.save_train_graph(self.train_graph)

    def graph_partition(self):
        if self.args['is_partition']:
            self.logger.info('graph partitioning')

            start_time = time.time()
            partition = GraphPartition(self.args, self.train_graph, self.data)
            self.community_to_node = partition.graph_partition()
            partition_time = time.time() - start_time
            self.logger.info("Partition cost %s seconds." % partition_time)
            self.data_store.save_community_data(self.community_to_node)
        else:
            self.community_to_node = self.data_store.load_community_data()

    def generate_shard_data(self):
        self.logger.info('generating shard data')

        self.shard_data = {}

        edge_index = self.data.edge_index  # shape: [2, num_edges]
        rating = self.data.rating if hasattr(self.data, 'rating') else None
        node_type = self.data.node_type  # 0=user, 1=item
        x_all = self.data.x if hasattr(self.data, 'x') else None

        for shard in range(self.args['num_shards']):
            shard_node_indices = torch.tensor(list(self.community_to_node[shard]), dtype=torch.long)

            shard_node_type = node_type[shard_node_indices]

            num_users = (shard_node_type == 1).sum().item()
            num_items = (shard_node_type == 0).sum().item()

            shard_node_indices = torch.tensor(
                sorted(self.community_to_node[shard]),
                dtype=torch.long
            )


            x = x_all[shard_node_indices] if x_all is not None else None

            old_to_new_id = {old.item(): new for new, old in enumerate(shard_node_indices)}
            new_to_old_id = {v: k for k, v in old_to_new_id.items()}

            src, dst = edge_index[0], edge_index[1]
            mask_src = torch.isin(src, shard_node_indices)
            mask_dst = torch.isin(dst, shard_node_indices)
            edge_mask = torch.logical_and(mask_src, mask_dst)
            shard_edge_index = edge_index[:, edge_mask]

            mapped_edge_index = shard_edge_index.clone()
            mapped_edge_index[0] = torch.tensor([old_to_new_id[n.item()] for n in shard_edge_index[0]])
            mapped_edge_index[1] = torch.tensor([old_to_new_id[n.item()] for n in shard_edge_index[1]])

            data_kwargs = {
                'edge_index': mapped_edge_index,
                'node_type': shard_node_type,
                'num_users': num_users,
                'num_items': num_items,
            }

            if x is not None:
                data_kwargs['x'] = x
            if rating is not None:
                data_kwargs['rating'] = rating[edge_mask]
            if hasattr(self.data, 'time'):
                data_kwargs['time'] = self.data.time[edge_mask]
            if hasattr(self.data, 'edge_type'):
                data_kwargs['edge_type'] = self.data.edge_type[edge_mask]

            if hasattr(self.data, 'edge_label_index'):
                edge_label_index = self.data.edge_label_index
                edge_label = self.data.edge_label if hasattr(self.data, 'edge_label') else None

                shard_node_set = set(shard_node_indices.tolist())

                valid_mask = [
                    i for i in range(edge_label_index.size(1))
                    if
                    edge_label_index[0, i].item() in shard_node_set and edge_label_index[1, i].item() in shard_node_set
                ]

                if valid_mask:
                    valid_mask = torch.tensor(valid_mask, dtype=torch.long)
                    filtered_edge_label_index = edge_label_index[:, valid_mask]

                    new_rows = []
                    new_cols = []
                    for i in range(filtered_edge_label_index.size(1)):
                        src = filtered_edge_label_index[0, i].item()
                        dst = filtered_edge_label_index[1, i].item()
                        new_rows.append(old_to_new_id[src])
                        new_cols.append(old_to_new_id[dst])

                    remapped_edge_label_index = torch.tensor([new_rows, new_cols], dtype=torch.long)
                    data_kwargs['edge_label_index'] = remapped_edge_label_index

                    if edge_label is not None:
                        data_kwargs['edge_label'] = edge_label[valid_mask]

            if not hasattr(self, 'shard_node_mappings'):
                self.shard_node_mappings = {}

            self.shard_node_mappings[shard] = new_to_old_id

            shard_data = Data(**data_kwargs)
            self.shard_data[shard] = shard_data
            print(self.shard_data[shard])

        self.data_store.save_shard_mappings(self.shard_node_mappings)
        self.data_store.save_shard_data(self.shard_data)

    def _ratio_delete_edges(self, edge_index):
        edge_index_np = edge_index.numpy()

        edges_set = set()
        for u, v in edge_index_np.T:
            edges_set.add(tuple(sorted((u, v))))
        print("没删除前无向边数量（去重后）:", len(edges_set))

        unique_edge_mask = edge_index_np[0] < edge_index_np[1]
        unique_indices = np.where(unique_edge_mask)[0]

        keep_num = int(len(unique_indices) * (1.0 - self.args['ratio_deleted_edges']))

        remain_unique_indices = np.random.choice(unique_indices, keep_num, replace=False)

        unique_indices_opposite = np.where(edge_index_np[0] > edge_index_np[1])[0]

        remain_opposite_indices = []
        for idx in remain_unique_indices:
            src = edge_index_np[0, idx]
            dst = edge_index_np[1, idx]
            opposite_idx = np.where((edge_index_np[0] == dst) & (edge_index_np[1] == src))[0]
            if len(opposite_idx) > 0:
                remain_opposite_indices.append(opposite_idx[0])

        remain_indices = np.concatenate([remain_unique_indices, np.array(remain_opposite_indices)])

        remain_indices = np.sort(remain_indices)

        return torch.from_numpy(edge_index_np[:, remain_indices]), remain_indices


