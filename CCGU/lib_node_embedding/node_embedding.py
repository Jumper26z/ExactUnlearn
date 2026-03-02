import logging
import torch
from torch_geometric.nn import LightGCN
from lib_dataset.data_store import DataStore


class NodeEmbedding:
    def __init__(self, args, graph, data):
        super(NodeEmbedding, self)

        self.logger = logging.getLogger(__name__)
        self.args = args
        self.graph = graph
        self.data = data

        self.data_store = DataStore(self.args)

    def sage_encoder(self):
        if self.args['is_gen_embedding']:
            self.logger.info("generating node embeddings with GraphSage...")

            node_to_embedding = {}
            # run sage
            # self.target_model = SAGE(self.data.num_features, len(self.data.y.unique()), self.data)
            self.target_model = LightGCN(
                num_nodes=self.data.num_nodes,
                embedding_dim=64,
                num_layers=2,
            ).to(self.args['cuda'])

            optimizer = torch.optim.Adam(self.target_model.parameters(), lr=0.001)

            self.target_model.train()
            for epoch in range(50):
                optimizer.zero_grad()
                out = self.target_model(self.data.edge_index.to(self.args['cuda']))
                loss = (out ** 2).sum()  # Dummy loss
                loss.backward()
                optimizer.step()

            self.target_model.eval()
            embeddings = self.target_model.get_embedding(self.data.edge_index.to(self.args['cuda']))
            embeddings = embeddings.detach().cpu().numpy()

            for node in self.graph.nodes:
                node_to_embedding[node] = embeddings[node]

            self.data_store.save_embeddings(node_to_embedding)
        else:
            node_to_embedding = self.data_store.load_embeddings()

        return node_to_embedding

