import torch
from torch_geometric.graphgym import cfg
from torch_geometric.graphgym.register import register_node_encoder


@register_node_encoder('LinearNode')
class LinearNodeEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()

        # self.encoder = torch.nn.Linear(cfg.share.dim_in - 1, emb_dim // 2)
        # self.emb_layer = torch.nn.Embedding(7, emb_dim // 2)
        # self.bn = torch.nn.BatchNorm1d(emb_dim // 2)
        self.encoder = torch.nn.Linear(cfg.share.dim_in, emb_dim)

    def forward(self, batch):
        # emb_res = self.emb_layer(batch.x[:, 0].long())
        # linear_res = self.encoder(batch.x[:, 1:].float())
        # linear_res = self.bn(linear_res)

        batch.x = self.encoder(batch.x.float())
        # batch.x = torch.cat([emb_res, linear_res], dim=1)
        return batch
