import torch.nn as nn
from torch import Tensor
from gssc.network.norm import NoneNorm, normdict
import warnings
from gssc.network import MaskedReduce
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import Sequential as PygSequential
from torch_geometric.utils import to_dense_batch


act_dict = {"relu": nn.ReLU(inplace=True), "ELU": nn.ELU(inplace=True), "silu": nn.SiLU(inplace=True), "softplus": nn.Softplus(), "softsign": nn.Softsign(), "softshrink": nn.Softshrink()}

class MLP(nn.Module):
    def __init__(self, indim: int, hiddim: int, outdim: int, numlayer: int=1, tailact: bool=True, dropout: float=0, norm: str="bn", activation: str="relu", tailbias=True, normparam: float=0.1) -> None:
        super().__init__()
        assert numlayer >= 0
        if isinstance(activation, str):
            activation = act_dict[activation]
        if isinstance(norm, str):
            norm = normdict[norm]
        if numlayer == 0:
            assert indim == outdim
            if norm != "none":
                warnings.warn("not equivalent to Identity")
                lin0 = nn.Sequential(norm(outdim, normparam))
            else:
                lin0 = nn.Sequential(NoneNorm())
        elif numlayer == 1:
            lin0 = nn.Sequential(nn.Linear(indim, outdim, bias=tailbias))
            if tailact:
                lin0.append(norm(outdim, normparam))
                if dropout > 0:
                    lin0.append(nn.Dropout(dropout, inplace=True))
                lin0.append(activation)
        else:
            lin0 = nn.Sequential(nn.Linear(hiddim, outdim, bias=tailbias))
            if tailact:
                lin0.append(norm(outdim, normparam))
                if dropout > 0:
                    lin0.append(nn.Dropout(dropout, inplace=True))
                lin0.append(activation)
            for _ in range(numlayer-2):
                lin0.insert(0, activation)
                if dropout > 0:
                    lin0.insert(0, nn.Dropout(dropout, inplace=True))
                lin0.insert(0, norm(hiddim, normparam))
                lin0.insert(0, nn.Linear(hiddim, hiddim))
            lin0.insert(0, activation)
            if dropout > 0:
                lin0.insert(0, nn.Dropout(dropout, inplace=True))
            lin0.insert(0, norm(hiddim, normparam))
            lin0.insert(0, nn.Linear(indim, hiddim))
        self.lin = lin0
        # self.reset_parameters()

    def forward(self, x: Tensor):
        return self.lin(x)


class PermEquiLayer(nn.Module):

    def __init__(self, hiddim: int, outdim: int, set2set: str, invout: bool, numlayers: int, **kwargs) -> None:
        super().__init__()
        assert set2set in ["deepset", "transformer"]
        if set2set == "deepset":
            self.set2set = PygSequential(
                "x, mask",
                [
                    (
                        Set2Set(hiddim, kwargs["combine"], kwargs["aggr"], res=kwargs["res"], **kwargs["mlpargs1"]),
                        "x, mask -> x",
                    )
                    for _ in range(numlayers)
                ]
                + [(nn.Identity(), "x -> x")],
            )
        elif set2set == "transformer":
            raise NotImplementedError
        if invout:
            self.set2vec = Set2Vec(hiddim, outdim, aggr=kwargs["pool"], **kwargs["mlpargs2"])
        else:
            self.set2vec = PygSequential("x, mask", [(MLP(hiddim, outdim, outdim, **kwargs["mlpargs2"]), "x->x")])

    def forward(self, x, mask):
        """
        x (B, N, d)
        mask (B, N)
        """
        # print(self.set2set)
        # print(torch.linalg.norm(x).item())
        x = self.set2set(x, mask)
        # print(torch.linalg.norm(x).item())
        x = self.set2vec(x, mask)
        # print(torch.linalg.norm(x).item())
        return x


class Set2Set(nn.Module):
    def __init__(self, hiddim: int, combine: str = "mul", aggr: str="sum", res: bool=True,  setdim: int=-2, **mlpargs) -> None:
        super().__init__()
        assert combine in  ["mul", "add"]
        self.mlp1 = MLP(hiddim, hiddim, hiddim, **mlpargs)
        self.mlp2 = MLP(hiddim, hiddim, hiddim, **mlpargs)
        self.setdim = setdim
        self.aggr = MaskedReduce.reduce_dict[aggr]
        self.res = res
        self.combine = combine

    def forward(self, x, mask):
        '''
        x (B, N, d)
        mask (B, N)
        '''
        x1 = self.mlp1(x)
        x1 = self.aggr(x1, mask.unsqueeze(-1), self.setdim).unsqueeze(self.setdim)
        x2 = self.mlp2(x)
        #print(torch.linalg.norm(x1).item(), torch.linalg.norm(x2).item())
        if self.combine == "mul":
            x1 = x1 * x2
        else:
            x1 = x1 + x2
        if self.res:
            x1 += x
        #print(torch.linalg.norm(x1).item())
        return x1


class Set2Vec(nn.Module):
    def __init__(self, hiddim: int, outdim: int, aggr: str="sum", setdim: int=-2, **mlpargs) -> None:
        super().__init__()
        self.mlp1 = MLP(hiddim, outdim, outdim, **mlpargs)
        self.mlp2 = MLP(outdim, outdim, outdim, **mlpargs)
        self.setdim = setdim
        self.aggr = MaskedReduce.reduce_dict[aggr]

    def forward(self, x, mask):
        '''
        x (B, N , d)
        mask (B, N)
        '''
        x1 = self.mlp1(x)
        x1 = self.aggr(x1, mask.unsqueeze(-1), self.setdim)
        return self.mlp2(x1)


class InitPEs(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lin = MLP(1, 2 * dim, dim, dropout=0.0, activation="silu", numlayer=1, norm="ln")

        config_dict = {
            "numlayers": 3,
            "nhead": 1,
            "dffn": 32,
            "norm_first": False,
            "aggr": "mean",
            "combine": "mul",
            "res": True,
            "mlpargs1": {
                "numlayer": 2,
                "norm": "ln",
                "tailact": True,
                "dropout": 0.0,
                "activation": "silu",
            },
            "mlpargs2": {
                "numlayer": 0,
                "norm": "none",
                "tailact": False,
                "dropout": 0.0,
                "activation": "silu",
            },
        }
        self.LambdaEncoder = PermEquiLayer(dim, dim, "deepset", False, **config_dict)

    def forward(self, batch):
        EigVals, nodemask = to_dense_batch(batch.EigVals, batch.batch)
        EigVecs, _ = to_dense_batch(batch.EigVecs, batch.batch)

        Lambda = EigVals[:, 0].squeeze(-1)[:, 1:]
        U = EigVecs[:, :, 1:]

        Lambda[Lambda.isnan()] = 0.0
        U[U.isnan()] = 0.0
        Lambdamask = torch.abs(Lambda) < 0.0001
        Lambda = torch.sqrt(Lambda)

        U.masked_fill_(Lambdamask.unsqueeze(1), 0)
        U.masked_fill_(~nodemask.unsqueeze(-1), 0)

        LambdaEmb = self.lin(Lambda.unsqueeze(-1))

        LambdaEmb = self.LambdaEncoder(LambdaEmb, Lambdamask)
        LambdaEmb = torch.where(Lambdamask.unsqueeze(-1), 0, LambdaEmb)

        init_pe = torch.einsum("bnm,bmd->bnmd", U, LambdaEmb)  # (#graph, N, M, d)
        batch.init_pe = init_pe
        return batch
