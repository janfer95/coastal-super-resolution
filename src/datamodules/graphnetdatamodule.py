import torch
import copy
import numpy as np
import lightning as L

import src.meshtools as mt

from pathlib import Path
from typing import Callable, Literal, Optional
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.data.separate import separate


class SwanDataModule(L.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 16,
        num_workers: int = 6,
        root_dir: str = "data",
        nregion: Literal[1, 2, 3] = 1,
        transform: Optional[Callable] = None,
    ):
        """SwanDataModule that loads the geometric Swan Data Sets

        Parameters
        ----------
        batch_size : int, optional
            Batch size of data set, by default 16.
        num_workers : int, optional
            Number of cpus used for data loader, by default 6.
        root_dir : str, optional
            Root directory where data is stored, by default "data".
        nregion : Literal[1, 2, 3], optional
            Which of the three regions to load, by default 1.
        transform : Optional[Callable], optional
            Data transforms after the dataset is created, by default None.
        """
        super().__init__()
        self.save_hyperparameters()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.nregion = nregion
        self.transform = transform

    def setup(self, stage: str):
        if stage == "fit":
            dataset_full = DataSet(
                root_dir=self.root_dir,
                nregion=self.nregion,
                train=True,
                transform=self.transform,
            )
            self.dataset_train, self.dataset_val = random_split(
                dataset_full, [0.8, 0.2]
            )

        if stage == "test" or stage == "predict":
            self.dataset_test = DataSet(
                root_dir=self.root_dir,
                nregion=self.nregion,
                train=False,
                transform=self.transform,
            )

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            follow_batch=["x_h", "x_l"],
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            follow_batch=["x_h", "x_l"],
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            follow_batch=["x_h", "x_l"],
        )

    def predict_dataloader(self):
        return DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            follow_batch=["x_h", "x_l"],
        )


class PairData(Data):
    """
    Class to natively hold high- and low-resolution graphs and
    edge indices. Supports batching over both resolutions.

    For more details refer to:
    https://pytorch-geometric.readthedocs.io/en/latest/advanced/batching.html
    """

    def __inc__(self, key, value, *args, **kwargs):
        # To batch edge indices correctly
        if key == "edge_index_h":
            return self.x_h.size(0)
        if key == "edge_index_l":
            return self.x_l.size(0)
        return super().__inc__(key, value, *args, **kwargs)


class DataSet(InMemoryDataset):
    def __init__(
        self,
        root_dir: str = "data",
        nregion: Literal[1, 2, 3] = 1,
        train: bool = True,
        transform: Optional[Callable] = None,
    ):
        """Constructor of the data set.

        Can be called without arguments, which defaults
        to most common use case.

        Parameters
        ----------
        root_dir : str
            Directory where all data is stored.
        nregion : Literal[1, 2, 3], optional
            Which of the three regions to load, by default 1.
        train : bool, optional
            If true, the training instead of test data is loaded,
            by default True.
        transform : Optional[Callable], optional
            Data transforms after the dataset is created, by default None.
        """
        self.root_dir = Path(root_dir)
        self.nregion = nregion
        if train:
            self.data_dir = self.root_dir / "training-data"
        else:
            self.data_dir = self.root_dir / "test-data"

        self.hs_path = self.data_dir / "hs"
        self.dir_path = self.data_dir / "dir"

        (
            self.pos_h,
            self.edge_index_h,
            self.edge_attrs_h,
        ) = self.__get_mesh_data("hr")
        (
            self.pos_l,
            self.edge_index_l,
            self.edge_attrs_l,
        ) = self.__get_mesh_data("lr")

        super().__init__("", transform)
        # This line is needed for InMemory DataSets
        self.data, self.slices = torch.load(self.processed_file_names[0])

    @property
    def raw_file_names(self) -> list[Path]:
        """List of file names of low-resolution data.

        Returns
        -------
        list[str]
            List of file names of low-resolution data.

        """
        return [self.hs_path / f"lr-data-region{self.nregion}.npy"]

    @property
    def processed_file_names(self):
        return [self.root_dir / "processed.pt"]

    def __get_mesh_data(self, res: Literal["hr", "lr"]):
        """Get list of coordinates, edge_indices, and edge_weights from
        specified mesh type for all regions.

        Parameters
        ----------
        res : Literal["hr", "lr"]
            One of "hr" or "lr" for high- or low-resolution mesh

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            Tuple of coordinates, edge indices and weights
        """

        mesh_path = self.root_dir / "mesh-data" / f"{res}-region{self.nregion}"
        mesh = mt.load_mesh(mesh_path)

        pos = torch.from_numpy(mesh.pos).float()

        edge_index = torch.from_numpy(mesh.edges).contiguous()
        row, col = edge_index

        dist = pos[row] - pos[col]
        edge_weight = torch.norm(dist, p=2, dim=-1).float().unsqueeze(-1)

        max_dist = edge_weight.max()

        edge_attrs = torch.cat([edge_weight, dist], dim=-1)
        # Normalize weights and relative coordinates
        edge_attrs = edge_attrs / max_dist

        return pos, edge_index, edge_attrs

    def delete_nodes(self, res="hr"):
        # Delete all nan nodes and remove corresponding edges
        if res == "lr":
            suffix = "_l"
        else:
            suffix = "_h"

        fname = (self.root_dir / "training-data"
                 / f"{res}-nan-mask-region{self.nregion}.npy")

        pos = "pos" + suffix
        edge_index = "edge_index" + suffix
        edge_attrs = "edge_attrs" + suffix

        nan_mask = torch.from_numpy(np.load(fname))

        # Delete nan nodes of bathymetry and coordinates
        setattr(self, pos, getattr(self, pos)[nan_mask])

        keep_nodes = nan_mask.nonzero().flatten()
        nan_edge_mask = torch.isin(
            getattr(self, edge_index), keep_nodes
        ).all(dim=0)

        # Delete edges with deleted nodes
        # But edges are now not correctly numbered!
        new_edges = getattr(self, edge_index)[:, nan_edge_mask]

        # Re-number
        idx = torch.arange(len(keep_nodes))
        # Create array that has non-zero entries at old indices
        mapping_array = torch.zeros(keep_nodes.max() + 1, dtype=torch.long)
        mapping_array[keep_nodes] = idx

        new_edges = mapping_array[new_edges]

        setattr(
            self,
            edge_index,
            new_edges
        )
        setattr(
            self,
            edge_attrs,
            getattr(self, edge_attrs)[nan_edge_mask]
        )

        return nan_mask.numpy()

    def process(self):
        """Get and save list of PairData objects.

        Returns
        -------
        list[PairData, ...]
            List of PairData Objects that contain all necessary inputs for
            training.

        """
        print("Normalizing...")
        region = f"region{self.nregion}"

        # Load low- and high-resolution data
        lr_hs = np.load(self.hs_path / f"lr-data-{region}.npy")
        lr_dir = np.load(self.dir_path / f"lr-data-{region}.npy")
        hr = np.load(self.hs_path / f"hr-data-{region}.npy")

        # Load nan indices and delete them
        lr_nan_mask = self.delete_nodes("lr")
        hr_nan_mask = self.delete_nodes("hr")

        # Remove nan data
        lr_hs = lr_hs[:, lr_nan_mask]
        lr_dir = lr_dir[:, lr_nan_mask]
        hr = hr[:, hr_nan_mask]

        # Load normalization data
        lr_hs_mean, lr_hs_std = np.load(
            self.hs_path / f"lr-znorm-{region}.npy"
        )
        lr_dir_mean, lr_dir_std = np.load(
            self.dir_path / f"lr-znorm-{region}.npy"
        )
        hr_mean, hr_std = np.load(self.hs_path / f"hr-znorm-{region}.npy")

        # Normalize
        lr_hs = (lr_hs - lr_hs_mean) / lr_hs_std
        lr_dir = (lr_dir - lr_dir_mean) / lr_dir_std
        hr = (hr - hr_mean) / hr_std

        # Convert to torch tensors
        lr_hs = torch.from_numpy(lr_hs).float().unsqueeze(-1)
        lr_dir = torch.from_numpy(lr_dir).float()
        lr = torch.cat([lr_hs, lr_dir], dim=-1)

        hr = torch.from_numpy(hr).float().unsqueeze(-1)

        print("Low-Resolution Input Shape", lr.shape)
        print("High-Resolution Output Shape", hr.shape)

        # Loop over all time steps and create list of PairData objects
        data_list = []
        for idx in range(lr.shape[0]):
            pairdata = PairData(x=lr[idx], y=hr[idx])
            data_list.append(pairdata)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_file_names[0])

    def get(self, idx: int) -> PairData:
        # Overwrite get function to avoid loading mesh data repeatedly
        if self.len() == 1:
            return copy.copy(self._data)

        if not hasattr(self, "_data_list") or self._data_list is None:
            self._data_list = self.len() * [None]
        elif self._data_list[idx] is not None:
            return copy.copy(self._data_list[idx])

        data = separate(
            cls=self._data.__class__,
            batch=self._data,
            idx=idx,
            slice_dict=self.slices,
            decrement=False,
        )

        self._data_list[idx] = copy.copy(data)

        pairdata = PairData(
            x_l=data.x,
            x_h=self.pos_h,  # Dummy data to batch variables correctly
            y=data.y,
            edge_index_h=self.edge_index_h,
            edge_index_l=self.edge_index_l,
            edge_attrs_h=self.edge_attrs_h,
            edge_attrs_l=self.edge_attrs_l,
            pos_h=self.pos_h,
            pos_l=self.pos_l,
        )

        return pairdata
