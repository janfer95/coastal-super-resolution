import torch

from lightning.pytorch.callbacks import BasePredictionWriter
from pathlib import Path


class PredictionWriter(BasePredictionWriter):
    def __init__(self, output_dir, write_interval):
        super().__init__(write_interval)
        self.output_dir = output_dir

    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        prediction,
        batch_indices,
        batch,
        batch_idx,
        dataloader_idx,
    ):
        fpath = Path(self.output_dir) / dataloader_idx / f"{batch_idx}.pt"
        torch.save(prediction, fpath)

    def write_on_epoch_end(self, trainer, pl_module, predictions,
                           batch_indices):
        fpath = Path(self.output_dir) / "Predictions" / "predictions.pt"
        fpath.parent.mkdir(exist_ok=True, parents=True)
        torch.save(torch.cat(predictions), fpath)
