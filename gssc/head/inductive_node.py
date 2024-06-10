import torch.nn as nn
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.layer import new_layer_config, MLP
from torch_geometric.graphgym.register import register_head
import torch


@register_head('inductive_node')
class GNNInductiveNodeHead(nn.Module):
    """
    GNN prediction head for inductive node prediction tasks.

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension. For binary prediction, dim_out=1.
    """

    def __init__(self, dim_in, dim_out):
        super(GNNInductiveNodeHead, self).__init__()
        self.layer_post_mp = MLP(
            new_layer_config(dim_in, dim_out, cfg.gnn.layers_post_mp,
                             has_act=False, has_bias=True, cfg=cfg))
        if cfg.extra.jk:
            self.jk_mlp = nn.Sequential(nn.Linear(cfg.gnn.dim_inner*cfg.gt.layers, cfg.gnn.dim_inner), nn.SiLU(), nn.Linear(cfg.gnn.dim_inner, cfg.gnn.dim_inner))

    def _apply_index(self, batch):
        return batch.x, batch.y

    def forward(self, batch):
        if cfg.extra.jk:
            batch.x = self.jk_mlp(torch.cat(batch.all_x, dim=-1))
        batch = self.layer_post_mp(batch)
        pred, label = self._apply_index(batch)
        return pred, label
