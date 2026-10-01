"""Logical launch verification for a teaching portability project.
It verifies a finite index map, NOT GPU instructions, races or device support.
"""
from dataclasses import dataclass
from collections import Counter
from typing import Callable


@dataclass(frozen=True)
class Launch:
    rows: int
    columns: int
    tile_rows: int = 16
    tile_columns: int = 16

    def __post_init__(self):
        values = (self.rows, self.columns, self.tile_rows, self.tile_columns)
        if any(type(v) is not int for v in values):
            raise TypeError("launch dimensions must be integers")
        if min(self.rows, self.columns) < 0 or min(self.tile_rows, self.tile_columns) <= 0:
            raise ValueError("negative extent or nonpositive tile")
        if self.rows * self.columns > 1_000_000:
            raise ValueError("finite verifier limited to one million outputs")
        if self.tile_rows * self.tile_columns > 4096:
            raise ValueError("finite verifier tile limit exceeded")

    @property
    def grid(self):
        return ((self.rows + self.tile_rows - 1) // self.tile_rows,
                (self.columns + self.tile_columns - 1) // self.tile_columns)


def row_major(row: int, column: int, launch: Launch) -> int:
    return row * launch.columns + column


def verify(launch: Launch,
           address: Callable[[int, int, Launch], int] = row_major) -> dict:
    """Enumerate logical participants, mask edges and verify output ownership.
    Each valid logical output must have exactly one writer. An out-of-range
    address is a failure even when another address happens to be missing.
    """
    counts = Counter()
    masked = 0
    outside = []
    gy, gx = launch.grid
    for by in range(gy):
        for bx in range(gx):
            for ty in range(launch.tile_rows):
                for tx in range(launch.tile_columns):
                    row = by * launch.tile_rows + ty
                    column = bx * launch.tile_columns + tx
                    if row >= launch.rows or column >= launch.columns:
                        masked += 1
                        continue
                    index = address(row, column, launch)
                    if type(index) is not int:
                        raise TypeError("address map must return an integer")
                    if not 0 <= index < launch.rows * launch.columns:
                        outside.append((row, column, index))
                    else:
                        counts[index] += 1
    missing = [i for i in range(launch.rows * launch.columns) if counts[i] == 0]
    duplicates = {i: n for i, n in counts.items() if n > 1}
    ok = not (missing or duplicates or outside)
    return {"status": "LOGICAL_MAP_VALIDATED" if ok else "FAILED",
            "rows": launch.rows, "columns": launch.columns,
            "tile": [launch.tile_rows, launch.tile_columns],
            "grid": list(launch.grid), "valid_writes": sum(counts.values()),
            "masked_participants": masked, "missing": missing,
            "duplicate_writes": duplicates, "out_of_range": outside,
            "scope": "finite logical output map only; no device execution or race proof"}
