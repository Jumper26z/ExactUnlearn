import torch
from torch_geometric.nn import LightGCN
import torch.utils.data
from torch_geometric.utils import degree
from tqdm import tqdm
import logging
from lib_gnn_model.gnn_base import GNNBase
import math

class LightGCN_run(GNNBase):
    def __init__(self, num_users, num_items, hidden_channels, num_layers):
        super().__init__()
        self.logger = logging.getLogger('lightgcn')
        self.num_users = num_users
        self.num_items = num_items
        self.num_nodes = self.num_users + self.num_items
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.model = LightGCN(
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
        mask = self.edge_index[0] < self.edge_index[1]
        self.train_edge_label_index = self.edge_index[:, mask] #按从小到大排序（小的是item,大的是user）
        train_loader = torch.utils.data.DataLoader(
            range(self.train_edge_label_index.size(1)),
            shuffle=True,
            batch_size=self.batch_size,
        )
        for epoch in range(epochs):
            total_loss = total_examples = 0
            for index in tqdm(train_loader, desc=f"Epoch {epoch}"):
                pos_edge_label_index = self.train_edge_label_index[:, index]
                neg_edge_label_index = torch.stack([
                    pos_edge_label_index[0],
                    torch.randint(0, self.num_items, (index.numel(),), device=self.device)
                ], dim=0)
                edge_label_index = torch.cat([pos_edge_label_index, neg_edge_label_index], dim=1)
                self.optimizer.zero_grad()
                pos_rank, neg_rank = self.model(self.edge_index, edge_label_index).chunk(2)
                loss = self.model.recommendation_loss(
                    pos_rank, neg_rank, node_id=edge_label_index.unique()
                )
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


    def test(self,k: int):

        emb = self.model.get_embedding(self.data.edge_index)
        item_emb, user_emb = emb[:self.num_items], emb[self.num_items:]

        precision = recall = mrr = ndcg = total_examples = 0
        for start in range(0, self.num_users, self.batch_size):
            end = min(start + self.batch_size, self.num_users)

            logits = user_emb[start:end] @ item_emb.t()

            # 训练边中用户节点的编号范围调整：
            mask = ((self.train_edge_label_index[0] >= self.num_items + start) &
                    (self.train_edge_label_index[0] < self.num_items + end))

            logits[self.train_edge_label_index[0, mask] - self.num_items - start,
            self.train_edge_label_index[1, mask]] = float('-inf')

            # ground_truth中用户节点的编号范围也要对应修改
            mask = ((self.data.edge_label_index[0] >= self.num_items + start) &
                    (self.data.edge_label_index[0] < self.num_items + end))

            ground_truth = torch.zeros_like(logits, dtype=torch.bool)
            ground_truth[self.data.edge_label_index[0, mask] - self.num_items - start,
            self.data.edge_label_index[1, mask]] = True

            # 下面 node_count 计算时的节点编号也一样减去 num_items + start
            node_count = degree(self.data.edge_label_index[0, mask] - self.num_items - start,
                                num_nodes=logits.size(0))

            topk_index = logits.topk(k, dim=-1).indices
            isin_mat = ground_truth.gather(1, topk_index)

            batch_valid_users = (node_count > 0).sum().item()
            if batch_valid_users == 0:
                continue

            precision += float((isin_mat.sum(dim=-1) / k).sum())
            recall += float((isin_mat.sum(dim=-1) / node_count.clamp(1e-6)).sum())
            total_examples += int((node_count > 0).sum())
            # 计算 MRR
            ranks = torch.where(isin_mat)[1] + 1 if isin_mat.any() else torch.tensor([])
            if len(ranks) > 0:
                mrr += float((1.0 / ranks.float()).sum())

            # 计算 NDCG
            ndcg_scores = []
            for gt, pred in zip(ground_truth, topk_index):
                ideal_dcg = sum([1.0 / math.log2(i + 2) for i in range(min(int(gt.sum().item()), k))])
                dcg = sum([(1.0 / math.log2(i + 2)) if gt[item] else 0.0 for i, item in enumerate(pred)])
                ndcg_scores.append(dcg / ideal_dcg if ideal_dcg > 0 else 0.0)
            ndcg += sum(ndcg_scores)

        return  precision / total_examples, recall / total_examples, mrr / total_examples, ndcg / total_examples

    def prepare_for_inference(self):
        """在加载模型后调用，准备推理所需的索引数据"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data = self.data.to(self.device)
        self.edge_index = self.data.edge_index
        mask = self.edge_index[0] < self.edge_index[1]
        self.train_edge_label_index = self.edge_index[:, mask]

    def posterior(self):
        self.model.eval()
        self.model, self.data = self.model.to(self.device), self.data.to(self.device)

        with torch.no_grad():
            emb = self.model.get_embedding(self.data.edge_index)
            item_emb, user_emb = emb[:self.num_items], emb[self.num_items:]
            logits = user_emb @ item_emb.t()

            train_u = self.train_edge_label_index[1] - self.num_items
            train_i = self.train_edge_label_index[0]
            logits[train_u, train_i] = float('-inf')

            posterior = torch.sigmoid(logits)

        return posterior.cpu()
