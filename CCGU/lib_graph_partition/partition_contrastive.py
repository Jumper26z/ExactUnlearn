import math
import logging
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from lib_graph_partition.partition import Partition
from lib_node_embedding.node_embedding import NodeEmbedding

class PartitionContrastive(Partition):
    def __init__(self, args, graph, dataset):
        super(PartitionContrastive, self).__init__(args, graph, dataset)

        self.logger = logging.getLogger('partition_contrastive')
        self.load_embeddings()

        self.temperature = args.get("contrast_tau", 0.5)
        self.contrast_epochs = args.get("contrast_epochs", 50)
        self.lr = args.get("contrast_lr", 1e-3)

    def load_embeddings(self):
        node_embedding = NodeEmbedding(self.args, self.graph, self.dataset)

        if self.partition_method == "sage_contrast":
            self.node_to_embedding = node_embedding.sage_encoder()
        else:
            raise Exception("unsupported embedding method")

    # ----------- InfoNCE Loss -----------
    def info_nce_loss(self, z1, z2):
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        sim_matrix = torch.matmul(z1, z2.T) / self.temperature
        labels = torch.arange(z1.size(0)).to(z1.device)

        loss = F.cross_entropy(sim_matrix, labels)
        return loss

    # ----------- 对比训练 prototype -----------
    def contrastive_training(self, embedding):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        z = torch.tensor(embedding, dtype=torch.float32).to(device)

        # prototype 层
        prototype = nn.Linear(z.size(1), self.num_shards, bias=False).to(device)

        optimizer = torch.optim.Adam(prototype.parameters(), lr=self.lr)

        for epoch in range(self.contrast_epochs):

            # 构造双视图（dropout增强）
            z1 = F.dropout(z, p=0.2, training=True)
            z2 = F.dropout(z, p=0.2, training=True)

            p1 = prototype(z1)
            p2 = prototype(z2)

            loss = self.info_nce_loss(p1, p2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                self.logger.info(f"Contrast Epoch {epoch}, Loss {loss.item():.4f}")

        return prototype

    def partition(self):

        self.logger.info("contrastive partitioning")

        node_ids = list(self.node_to_embedding.keys())
        embedding = np.array(
            [self.node_to_embedding[n] for n in node_ids]
        )

        # -------- Stage 1: 对比训练 --------
        prototype = self.contrastive_training(embedding)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        z = torch.tensor(embedding, dtype=torch.float32).to(device)

        with torch.no_grad():
            logits = prototype(z)
            probs = F.softmax(logits, dim=1)
            cluster_labels = torch.argmax(probs, dim=1).cpu().numpy()

        if not self.args["is_constrained"]:
            community_to_node = {
                i: np.where(cluster_labels == i)[0]
                for i in range(self.num_shards)
            }
        else:
            node_threshold = math.ceil(
                self.graph.number_of_nodes() / self.num_shards
                + self.args["shard_size_delta"] *
                (self.graph.number_of_nodes()
                 - self.graph.number_of_nodes() / self.num_shards)
            )

            self.logger.info(
                f"#.nodes: {self.graph.number_of_nodes()}, "
                f"Shard threshold: {node_threshold}"
            )

            community_to_node = {i: [] for i in range(self.num_shards)}

            # 根据 softmax 概率排序分配
            confidence = probs.max(dim=1).values.cpu().numpy()
            sorted_indices = np.argsort(-confidence)

            for idx in sorted_indices:
                sorted_clusters = np.argsort(
                    -probs[idx].cpu().numpy()
                )
                for cid in sorted_clusters:
                    if len(community_to_node[cid]) < node_threshold:
                        community_to_node[cid].append(idx)
                        break

            for i in range(self.num_shards):
                community_to_node[i] = np.array(
                    community_to_node[i]
                )

        return community_to_node