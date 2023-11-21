import logging
from pathlib import Path
from typing import Optional

import typer

from ._objects import Distribution, State, Patient, Landscape
from ._misc import md5_hexdigest


logging.basicConfig()
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

RESULT_FILENAME = "syn_{_hash}.csv"
TYPER = typer.Typer(help="CatSyn - A tool for synthesising categorical datasets.")


@TYPER.command()
def syn(
    config: Path = typer.Option(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    ),
    save_dir: Optional[Path] = typer.Option(
        None,
        exists=True,
        file_okay=False,
        dir_okay=True,
        writable=True,
        readable=True,
        resolve_path=True,
    ),
    save_graph: Optional[bool] = typer.Option(False, "--graph", "-g"),
    force: Optional[bool] = typer.Option(False, "--force", "-f"),
) -> Path:
    """
    Synthesise a dataset from a config.
    """
    result_path = Path(
        RESULT_FILENAME.format(_hash=md5_hexdigest(open(config, "r").read())[:7])
    )
    if save_dir is not None:
        result_path = Path(save_dir) / result_path
    if result_path.exists() and not force:
        raise ValueError(
            f"Found dataset at {result_path.absolute()}. This means you've already synthesised this config."
        )
    ls = Landscape(config_path=config)
    if save_graph:
        LOGGER.info(f"State graph saved to: {result_path.parent.absolute() / 'graph.png'}")
        ls.draw_graph()
        ls.save_adjacenct_matrix()
    LOGGER.info(f"Synthesising {ls._events} events.")
    df = ls.syn()
    df.to_csv(result_path, index=False)
    LOGGER.info(f"Synthetic dataset saved to: {result_path.absolute()}")

    return result_path

def ephemeral_syn(config):
    ls = Landscape(config_path=config)
    LOGGER.info(f"Synthesising {ls._events} events.")
    df = ls.syn()
    return df

if __name__ == "__main__":
    TYPER()
