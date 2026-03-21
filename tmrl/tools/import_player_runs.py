"""Import recorded player runs into replay dataset data.pkl."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import tyro
from loguru import logger

import tmrl.config as cfg
import tmrl.config.config_objects as cfg_obj
from tmrl.tools.player_runs import import_player_runs_to_dataset


@dataclass
class ImportPlayerRunsCli:
    """CLI for importing one or more player-run files."""

    paths: str
    """Comma-separated list of .pkl player-run files."""

    overwrite: bool = False
    """Overwrite existing dataset instead of appending."""

    max_samples: int = 0
    """Max raw samples to keep in resulting dataset (0 keeps all)."""

    dry_run: bool = False
    """Validate/convert only; don't write dataset file."""


def _parse_paths(paths: str) -> list[str]:
    return [p.strip() for p in paths.split(",") if p.strip()]


def import_player_runs(
    *,
    paths: list[str],
    overwrite: bool = False,
    max_samples: int | None = None,
    dry_run: bool = False,
    dataset_path: str | None = None,
) -> dict:
    """Import player-run files to the configured replay dataset."""
    if not paths:
        raise ValueError("At least one player-run path is required.")
    for p in paths:
        if not Path(p).is_file():
            raise FileNotFoundError(f"Run file not found: {p}")

    target_dataset_path = dataset_path or cfg.DATASET_PATH
    result = import_player_runs_to_dataset(
        paths,
        memory_factory=cfg_obj.MEMORY,
        dataset_path=target_dataset_path,
        overwrite=overwrite,
        max_samples=max_samples,
        dry_run=dry_run,
    )
    logger.info(
        "Imported {} file(s), {} samples into '{}'. dry_run={} trimmed={}",
        result["imported_files"],
        result["imported_samples"],
        result["dataset_file"],
        result["dry_run"],
        result["trimmed_raw_samples"],
    )
    return result


if __name__ == "__main__":
    cli = tyro.cli(ImportPlayerRunsCli)
    import_player_runs(
        paths=_parse_paths(cli.paths),
        overwrite=cli.overwrite,
        max_samples=None if cli.max_samples <= 0 else cli.max_samples,
        dry_run=cli.dry_run,
    )
