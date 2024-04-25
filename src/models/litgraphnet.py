import torch
import torch.nn.functional as F
import lightning as L

from torch_geometric.nn import MLP
from torch_geometric.nn.unpool import knn_interpolate

from ..datamodules.graphnetdatamodule import PairData
from .graphnetblock import GNBlock


class BaseGnnModule(L.LightningModule):
    def __init__(self):
        super().__init__()

    def forward(self):
        raise NotImplementedError

    def _shared_step(self, batch, step: str):
        # Get batch size for logging
        batch_size = batch.num_graphs

        pred = self(batch)
        l1_loss = F.l1_loss(pred, batch.y)
        l2_loss = F.mse_loss(pred, batch.y)
        self.log(
            f"{step}_loss",
            l1_loss,
            on_epoch=True,
            batch_size=batch_size,
            prog_bar=True,
            logger=True,
        )
        self.log(
            f"{step}_l2_loss",
            l2_loss,
            on_epoch=True,
            batch_size=batch_size,
            prog_bar=True,
            logger=True,
        )
        return {"loss": l1_loss}

    def training_step(self, train_batch, batch_idx):
        return self._shared_step(train_batch, "train")

    def validation_step(self, val_batch, batch_idx):
        return self._shared_step(val_batch, "val")

    def test_step(self, test_batch, batch_idx):
        return self._shared_step(test_batch, "test")


class LitGraphNet(BaseGnnModule):
    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 256,
        num_lr_layers: int = 5,
        num_hr_layers: int = 5,
        k: int = 3,
    ):
        """Initiate KnnGnnRes model

        Parameters
        ----------
        in_channels : int, optional
            Number of features / input channels, by default 1
        hidden_channels : int, optional
            Number of hidden channels, by default 256
        num_lr_layers : int, optional
            Number of low-resolution layers
        num_hr_layers : int, optional
            Number of high-resolution layers
        k : int, optional
            Number of nearest neighbors in knn upsampling, by default 3
        """
        super().__init__()
        self.save_hyperparameters()

        self.k = k
        self.num_lr_layers = num_lr_layers
        self.num_hr_layers = num_hr_layers

        chin = in_channels
        chh = hidden_channels

        # Encoders
        self.lr_encoder = MLP([chin, chh, chh],
                              norm=None, plain_last=False, act="silu")
        self.hr_edge_encoder = MLP([3, chh, chh],
                                   norm=None, plain_last=False, act="silu")
        self.lr_edge_encoder = MLP([3, chh, chh],
                                   norm=None, plain_last=False, act="silu")

        node_mlp_layers = (2 * chh, chh, chh)
        edge_mlp_layers = (3 * chh, chh, chh)

        # Low-Resolution Blocks
        self.lr_convs = torch.nn.ModuleList()
        for i in range(self.num_lr_layers):
            self.lr_convs.append(GNBlock(node_mlp_layers, edge_mlp_layers))

        # High-Resolution Blocks
        self.hr_convs = torch.nn.ModuleList()
        for i in range(self.num_hr_layers):
            self.hr_convs.append(GNBlock(node_mlp_layers, edge_mlp_layers))

        # Decoder
        self.hr_decoder = MLP([chh, chh, chh, 1], norm=None)

    def forward(self, data: PairData) -> torch.Tensor:
        x_l = data.x_l
        edge_index_h, edge_index_l = data.edge_index_h, data.edge_index_l
        edge_attrs_h, edge_attrs_l = data.edge_attrs_h, data.edge_attrs_l
        pos_h, pos_l = data.pos_h, data.pos_l

        # Encode features
        xl = self.lr_encoder(x_l)
        ehenc = self.hr_edge_encoder(edge_attrs_h)
        elenc = self.lr_edge_encoder(edge_attrs_l)

        for conv in self.lr_convs:
            xl, elenc = conv(xl, edge_index_l, elenc)

        xh = knn_interpolate(
            xl, pos_l, pos_h, data.x_l_batch, data.x_h_batch, k=self.k
            )
        xh = F.relu(xh)

        for conv in self.hr_convs:
            xh, elenc = conv(xh, edge_index_h, ehenc)

        xdec = self.hr_decoder(xh)

        return xdec
