import pickle
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.tri import Triangulation
from typing import Literal, Optional, Union


class Mesh:
    """Class to store and manipulate unstructured meshes.

    Mesh is a utility class to store and manipulate unstructured meshes.

    Parameters
    ----------
    pos : Optional[np.ndarray]
        Array with the node coordinates (x, y) of shape (nnodes, 2).
    bathymetry : Optional[np.ndarray]
        Array with the bathymetry at each node.
    triangles : Optional[np.ndarray]
        Array with the triangles of the mesh of shape (ntriangles, 3).
    edges : Optional[np.ndarray]
        Array with the edges of the triangles of shape (2, nedges).
        Can be passed explicitly, but is otherwise computed from `triangles`.
    edge_weights : Optional[np.ndarray]
        Array with the weights of the edges.
    name : str
        Name and or description of the mesh.

    Methods
    -------
    compute_edge_weights
        Compute the edge weights with a given normalization.
    save_mesh
        Save the mesh to a file.
    """
    def __init__(
        self,
        pos: Optional[np.ndarray] = None,
        bathymetry: Optional[np.ndarray] = None,
        triangles: Optional[np.ndarray] = None,
        edges: Optional[np.ndarray] = None,
        edge_weights: Optional[np.ndarray] = None,
        name: str = "Mesh0",
    ):
        self.name = name
        self.pos = pos
        self.bathymetry = bathymetry
        self.triangles = triangles
        self._edges = edges
        self.edge_weights = edge_weights

    def __repr__(self):
        message = f"{self.name}\n"
        message += f"Number of Nodes: {self.nnodes}\n"
        message += f"Number of Edges: {self.nedges}\n"
        message += f"Pos (Coordinates): {self.pos is not None}\n"
        message += f"Bathymetry: {self.bathymetry is not None}\n"
        message += f"Triangles: {self.triangles is not None}\n"
        message += f"Edges: {self.edges is not None}\n"
        message += f"Edge weights: {self.edge_weights is not None}\n"

        return message

    # -------------------------------------------------------------------------
    # Attributes
    @property
    def nnodes(self) -> Optional[int]:
        return None if self.pos is None else self.pos.shape[0]

    @property
    def ntriangles(self) -> Optional[int]:
        return None if self.triangles is None else self.triangles.shape[0]

    @property
    def ntri(self) -> Optional[int]:
        return self.ntriangles

    @property
    def nedges(self) -> Optional[int]:
        return None if self.edges is None else self.edges.shape[1]

    @property
    def edges(self) -> Optional[np.ndarray]:
        if self._edges is None:
            self._edges = self.__triangles_to_edges()
        return self._edges

    # -------------------------------------------------------------------------
    # Methods
    def __triangles_to_edges(self) -> np.ndarray:
        if self.triangles is None:
            return None

        edges = np.concatenate([self.triangles[:, [0, 1]],
                                self.triangles[:, [1, 2]],
                                self.triangles[:, [2, 0]]], axis=0,)
        edges = np.unique(np.r_[edges, np.fliplr(edges)], axis=0).T

        return edges

    def compute_edge_weights(
        self,
        normalization: Literal["none", "z", "max"] = "none"
    ) -> Optional[np.ndarray]:
        """Compute the edge weights with a given normalization.

        The edge weights are computed as the Euclidean distance between
        the nodes of the mesh. The normalization parameter can be used to
        standardize or normalize the edge weights.

        The edge weights are returned directly, as well as stored in the
        `edge_weights` attribute for later use.

        Parameters
        ----------
        normalization : Literal["none", "z", "max"]
            Normalization method to use. Options are:
            - "none": No normalization is applied. This is the default
            - "z": Standardize by the mean and standard deviation.
            - "max": Normalize by the maximum value.

        Returns
        -------
        np.ndarray
            Array with the edge weights.
        """
        if self.edges is None:
            self.edge_weights = None
            return None

        # Compute edge weights
        edge_weights = np.linalg.norm(
            self.pos[self.edges[0]] - self.pos[self.edges[1]], axis=1
        )
        if normalization == "z":
            edge_weights = (
                (edge_weights - edge_weights.mean()) / edge_weights.std()
            )
        elif normalization == "max":
            edge_weights = edge_weights / edge_weights.max()

        self.edge_weights = edge_weights
        return edge_weights

    def plot(
        self,
        data: Optional[np.ndarray] = None,
        ax: Optional[plt.Axes] = None,
        **kwargs,
    ) -> plt.Axes:
        """Plot some data on the mesh. Defaults to a bathymetry plot.

        Parameters
        ----------
        data : Optional[np.ndarray]
            Array with the data to plot. The length of the array should be
            the same as the number of nodes in the mesh. If not provided,
            the bathymetry is plotted.
        ax : Optional[plt.Axes]
            Matplotlib axis to plot the mesh on. If not provided, a new
            axis is created.
        **kwargs
            Additional keyword arguments to pass to meshtools.plot_mesh_data.
        """
        if data is None:
            if self.bathymetry is None:
                raise ValueError(
                    "No data provided and no bathymetry available."
                )
            data = self.bathymetry

        if ax is None:
            fig, ax = plt.subplots()

        ax = plot_mesh_data(data, self, ax, **kwargs)

        return ax

    def save_mesh(self, path: Union[str, Path]) -> None:
        """
        Save the mesh to a file.

        Parameters
        ----------
        path : Union[str, Path]
            Path to save the mesh. Extension is automatically added,
            if necessary.
        """
        path = Path(path).with_suffix(".pkl")

        with open(path, "wb") as f:
            pickle.dump(self, f)


def load_mesh(path: Union[str, Path]) -> Mesh:
    """
    Load a mesh from a file.

    Parameters
    ----------
    path : Union[str, Path]
        Path to load the mesh from. Suffix is automatically added,
        if necessary.

    Returns
    -------
    Mesh
        Mesh object with the loaded data.
    """
    path = Path(path).with_suffix(".pkl")

    with open(path, "rb") as f:
        mesh = pickle.load(f)

    return mesh


def plot_mesh_data(
    data: np.ndarray,
    mesh: Mesh,
    ax: plt.Axes,
    **kwargs,
) -> plt.Axes:
    """Plot data on the mesh.

    Parameters
    ----------
    data : np.ndarray
        Array with the data to plot. The length of the array should be
        the same as the number of nodes in the mesh.
    mesh : Mesh
        Mesh object that contains node coordinates (pos) and mesh triangles.
    ax : plt.Axes
        Matplotlib axis to plot the data on.
    **kwargs
        Additional keyword arguments to pass to `plt.tripcolor`.

    Returns
    -------
    plt.Axes
        Axes with the plot.
    """
    triangulation = Triangulation(
        mesh.pos[:, 0], mesh.pos[:, 1], triangles=mesh.triangles
    )

    nan_bool = np.isnan(data)
    tri_mask = np.any(nan_bool[triangulation.triangles], axis=1)
    triangulation.set_mask(tri_mask)

    data = np.nan_to_num(data)

    ax.tripcolor(triangulation, data, **kwargs)

    return ax
