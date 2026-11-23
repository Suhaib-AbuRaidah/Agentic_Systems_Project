#!/usr/bin/env python3
"""Run the planar linkage solver with `uv run python planar_linkage_synth.py`."""

from __future__ import annotations

from articraft.sdk._planar_synthesis import (
    PlanarModel,
    SpecificationError,
    fivebar_example,
    fourbar_example,
    main,
    schema,
    slidercrank_example,
    synthesize_planar_mechanism,
)

__all__ = [
    "PlanarModel",
    "SpecificationError",
    "fivebar_example",
    "fourbar_example",
    "main",
    "schema",
    "slidercrank_example",
    "synthesize_planar_mechanism",
]

if __name__ == "__main__":
    raise SystemExit(main())
