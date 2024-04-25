# Disable pyg developer warnings about TypedStorage
import warnings

from lightning.pytorch.cli import LightningCLI

warnings.filterwarnings(
    "ignore", category=UserWarning, message="TypedStorage is deprecated"
)


def call_cli():
    cli = LightningCLI(parser_kwargs={"parser_mode": "omegaconf"})  # noqa

    # tuner = Tuner(cli.trainer)
    # print(tuner.lr_find(cli.model, datamodule=cli.datamodule))


if __name__ == "__main__":
    call_cli()
    # os.system("shutdown")
