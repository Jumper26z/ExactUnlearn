import torch
import torch.utils.data
from torch_geometric.utils import degree
from torch_geometric.nn import GCNConv
import logging
from lib_gnn_model.gnn_base import GNNBase
from tqdm import tqdm
import math

class NGCF_run(GNNBase):
    def __init__(self, num_users, num_items, hidden_channels, num_layers):
        super().__init__()
        self.logger = logging.getLogger('ngcf')
        self.num_users = num_users
        self.num_items = num_items
        self.num_nodes = self.num_users + self.num_items
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.model = NGCF(
            num_nodes=self.num_nodes,
            embedding_dim=self.hidden_channels,
            num_layers=self.num_layers
        ).to(self.device)
        self.data = None

    def train_model(self, epochs=100, batch_size=64, k=10):
        self.data = self.data.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.device = self.data.x.device
        self.edge_index = self.data.edge_index
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.batch_size = batch_size
        self.model.train()

        # Prepare training data
        train_loader = torch.utils.data.DataLoader(
            range(self.num_users),
            shuffle=True,
            batch_size=self.batch_size,
        )

        for epoch in range(epochs):
            total_loss = total_examples = 0
            for index in tqdm(train_loader, desc=f"Epoch {epoch}"):
                pos_edge_label_index = self.edge_index[:, index]
                neg_edge_label_index = torch.stack([
                    pos_edge_label_index[0],
                    torch.randint(0, self.num_items, (index.numel(),), device=self.device)
                ], dim=0)
                edge_label_index = torch.cat([pos_edge_label_index, neg_edge_label_index], dim=1)

                self.optimizer.zero_grad()

                # Forward pass with NGCF model
                pos_rank, neg_rank = self.model(self.edge_index, edge_label_index).chunk(2)

                # Compute contrastive loss (use appropriate loss for NGCF)
                loss = self.model.contrastive_loss(pos_rank, neg_rank)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * pos_rank.size(0)
                total_examples += pos_rank.size(0)

            avg_loss = total_loss / total_examples
            print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")

        self.model.eval()
        with torch.no_grad():
            precision, recall, mrr, ndcg = self.test(k=k)
            print(f"Final Evaluation -> Precision@{k}: {precision:.4f}, Recall@{k}: {recall:.4f}, "
                  f"MRR@{k}: {mrr:.4f}, NDCG@{k}: {ndcg:.4f}")

    def test(self, k: int):
        emb = self.model.get_embedding(self.data.edge_index)
        item_emb, user_emb = emb[:self.num_items], emb[self.num_items:]

        precision = recall = mrr = ndcg = total_examples = 0
        for start in range(0, self.num_users, self.batch_size):
            end = min(start + self.batch_size, self.num_users)

            logits = user_emb[start:end] @ item_emb.t()

            # Training edge label adjustment for user nodes:
            mask = ((self.edge_index[0] >= self.num_items + start) &
                    (self.edge_index[0] < self.num_items + end))
            logits[self.edge_index[0, mask] - self.num_items - start,
            self.edge_index[1, mask]] = float('-inf')

            # Adjust ground_truth for corresponding user nodes in test
            ground_truth = torch.zeros_like(logits, dtype=torch.bool)
            ground_truth[self.data.edge_label_index[0, mask] - self.num_items - start,
            self.data.edge_label_index[1, mask]] = True

            node_count = degree(self.data.edge_label_index[0, mask], num_nodes=logits.size(0))

            topk_index = logits.topk(k, dim=-1).indices
            isin_mat = ground_truth.gather(1, topk_index)

            batch_valid_users = (node_count > 0).sum().item()
            if batch_valid_users == 0:
                continue

            precision += float((isin_mat.sum(dim=-1) / k).sum())
            recall += float((isin_mat.sum(dim=-1) / node_count.clamp(1e-6)).sum())
            total_examples += int((node_count > 0).sum())
            # Calculate MRR
            ranks = torch.where(isin_mat)[1] + 1 if isin_mat.any() else torch.tensor([])
            if len(ranks) > 0:
                mrr += float((1.0 / ranks.float()).sum())

            # Calculate NDCG
            ndcg_scores = []
            for gt, pred in zip(ground_truth, topk_index):
                ideal_dcg = sum([1.0 / math.log2(i + 2) for i in range(min(int(gt.sum().item()), k))])
                dcg = sum([(1.0 / math.log2(i + 2)) if gt[item] else 0.0 for i, item in enumerate(pred)])
                ndcg_scores.append(dcg / ideal_dcg if ideal_dcg > 0 else 0.0)
            ndcg += sum(ndcg_scores)

        return precision / total_examples, recall / total_examples, mrr / total_examples, ndcg / total_examples

    def prepare_for_inference(self):
        """Prepare for inference after loading model"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data = self.data.to(self.device)
        self.edge_index = self.data.edge_index

    def posterior(self):
        """Post-processing for inference"""
        self.model.eval()
        self.model, self.data = self.model.to(self.device), self.data.to(self.device)

        with torch.no_grad():
            emb = self.model.get_embedding(self.data.edge_index)
            item_emb, user_emb = emb[:self.num_items], emb[self.num_items:]
            logits = user_emb @ item_emb.t()

            train_u = self.train_edge_label_index[1] - self.num_items  # user node index mapping
            train_i = self.train_edge_label_index[0]  # item node index itself
            logits[train_u, train_i] = float('-inf')

            posterior = torch.sigmoid(logits)

        return posterior.cpu()


# NGCF model definition
class NGCF(torch.nn.Module):
    def __init__(self, num_nodes, embedding_dim, num_layers):
        super(NGCF, self).__init__()
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.conv_layers = torch.nn.ModuleList()
        self.activation = torch.nn.ReLU()

        for _ in range(num_layers):
            self.conv_layers.append(GCNConv(self.num_nodes, self.embedding_dim))

    def forward(self, edge_index, edge_label_index):
        embeddings = edge_index
        for conv in self.conv_layers:
            embeddings = conv(embeddings, edge_index)
            embeddings = self.activation(embeddings)  # Apply activation
        return embeddings

    def contrastive_loss(self, pos_rank, neg_rank):
        # Example of contrastive loss function (use a simple margin-based loss or any suitable one)
        margin = 1.0
        loss = torch.max(neg_rank - pos_rank + margin, torch.zeros_like(pos_rank))
        return loss.mean()

    def get_embedding(self, edge_index):
        # Retrieve learned embeddings for the nodes
        return self.conv_layers[-1](edge_index)