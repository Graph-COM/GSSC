import torch
import torch.nn as nn
from einops import rearrange
from torch_geometric.utils import to_dense_batch
import torch.nn.functional as F
from torch_geometric.graphgym.config import cfg


class GSSC(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.deg_coef = nn.Parameter(torch.zeros(1, 1, cfg.gnn.dim_inner, 2))
        nn.init.xavier_normal_(self.deg_coef)
        if cfg.extra.more_mapping:
            self.x_head_mapping = nn.Linear(cfg.gnn.dim_inner, cfg.gnn.dim_inner, bias=False)
            self.q_pe_head_mapping = nn.Linear(cfg.extra.init_pe_dim, cfg.extra.init_pe_dim, bias=False)
            self.k_pe_head_mapping = nn.Linear(cfg.extra.init_pe_dim, cfg.extra.init_pe_dim, bias=False)
            self.out_mapping = nn.Linear(cfg.gnn.dim_inner, cfg.gnn.dim_inner, bias=True)
        else:
            self.x_head_mapping = nn.Identity()
            self.q_pe_head_mapping = nn.Identity()
            self.k_pe_head_mapping = nn.Identity()
            self.out_mapping = nn.Identity()
        if cfg.extra.reweigh_self:
            self.reweigh_pe = nn.Linear(cfg.extra.init_pe_dim, cfg.extra.init_pe_dim, bias=False)
            self.reweigh_x = nn.Linear(cfg.gnn.dim_inner, cfg.gnn.dim_inner, bias=False)
            if cfg.extra.reweigh_self == 2:
                self.reweigh_pe_2 = nn.Linear(cfg.extra.init_pe_dim, cfg.extra.init_pe_dim, bias=False)

    def forward(self, x, batch):
        init_pe = batch.init_pe
        log_deg = to_dense_batch(batch.log_deg, batch.batch)[0][..., None]
        x = torch.stack([x, x * log_deg], dim=-1)
        x = (x * self.deg_coef).sum(dim=-1)

        if cfg.extra.reweigh_self:
            pe_reweigh = self.reweigh_pe(init_pe)
            x_reweigh = self.reweigh_x(x)
            pe_reweigh_2 = self.reweigh_pe_2(init_pe) if cfg.extra.reweigh_self == 2 else pe_reweigh

        x = self.x_head_mapping(x)
        q_pe = self.q_pe_head_mapping(init_pe)
        k_pe = self.k_pe_head_mapping(init_pe)
        first = torch.einsum("bnrd, bnl -> brdl", k_pe, x)
        x = torch.einsum("bnrd, brdl -> bnl", q_pe, first)
        x = self.out_mapping(x)

        if cfg.extra.reweigh_self:
            x = x + (pe_reweigh * pe_reweigh_2).sum(dim=(-1, -2))[..., None] * x_reweigh
        return x
