import torch

from torch.nn import Sequential
from torch_geometric.nn import MLP, LayerNorm
from torch_geometric.utils import scatter


class GNBlock(torch.nn.Module):
    def __init__(
        self,
        node_mlp_layers: tuple[int, ...] = (128, 128, 128),
        edge_mlp_layers: tuple[int, ...] = (128, 128, 128),
        aggr: str = "mean",
    ):
        super().__init__()
        self.node_mlp = MLP(
            node_mlp_layers, norm=None, plain_last=False, act="silu"
        )
        self.edge_mlp = MLP(
            edge_mlp_layers, norm=None, plain_last=False, act="silu"
        )
        # self.node_mlp = self._make_mlp(node_mlp_layers)
        # self.edge_mlp = self._make_mlp(edge_mlp_layers)
        self.aggr = aggr

    def reset_parameters(self):
        models = [self.node_mlp, self.edge_mlp]
        for model in models:
            if hasattr(model, "reset_parameters"):
                model.reset_parameters()

    def _make_mlp(self, mlp_layers):
        return Sequential(
            MLP(mlp_layers, norm=None, plain_last=True),
            LayerNorm(mlp_layers[-1])
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        row, col = edge_index
        # Update edges
        edge_attr = self.edge_mlp(
            torch.cat((edge_attr, x[row], x[col]), dim=-1)
            ) + edge_attr
        # Aggregate edges
        aggr_edges = scatter(
            edge_attr, col, dim=0, dim_size=x.size(0), reduce=self.aggr
        )
        # Update nodes
        x = self.node_mlp(torch.cat((aggr_edges, x), dim=-1)) + x
        return x, edge_attr
