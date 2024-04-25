import os

from torch import Tensor
from typing import Dict, Optional
from datetime import date
from lightning.pytorch.callbacks import ModelCheckpoint

class DateModelCheckpoint(ModelCheckpoint):
    """Same as ModelCheckpoint, but automatically adds date to filename
    """
    def format_checkpoint_name(self, metrics: Dict[str, Tensor],
                               filename: Optional[str] = None,
                               ver: Optional[int] = None) -> str:
        filename = filename or self.filename
        filename = self._format_checkpoint_name(filename, metrics, auto_insert_metric_name=self.auto_insert_metric_name)
        
        # Get current day
        today = date.today().strftime("%Y%m%d")

        if ver is not None:
            filename = self.CHECKPOINT_JOIN_CHAR.join((filename, f"v{ver}"))

        ckpt_name = f"{filename}{today}{self.FILE_EXTENSION}"
        return os.path.join(self.dirpath, ckpt_name) if self.dirpath else ckpt_name