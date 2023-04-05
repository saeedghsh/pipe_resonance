""""""

from typing import Tuple, NewType


Color = NewType('Color', Tuple[int, int, int])


class Pixel:
    def __init__(self, *, row: int = None, col: int = None, x: int = None, y: int = None) -> None:
        assert sum([row is not None, y is not None]) == 1, "row and y cannot be set simultaneously"
        assert sum([col is not None, x is not None]) == 1, "col and x cannot be set simultaneously"
        self._row = row if row is not None else y
        self._col = col if col is not None else x

    def __repr__(self) -> str:
        return f"row[y]={self.row},\tcol[x]={self.col}"

    @property
    def row(self) -> int:
        return self._row
    @property
    def y(self) -> int:
        return self.row
    @property
    def col(self) -> int:
        return self._col
    @property
    def x(self) -> int:
        return self.col
    
    @row.setter
    def row(self, row: int):
        self._row = row
    @row.setter
    def y(self, y: int):
        self.row = y
    @col.setter
    def col(self, col: int):
        self._col = col
    @x.setter
    def x(self, x: int):
        self.col = x

    def __add__(self, other: "Pixel") -> "Pixel":
        return Pixel(row=self.row+other.row, col=self.col+other.col)
    def __iadd__(self, other: "Pixel") -> "Pixel":
        return Pixel(row=self.row+other.row, col=self.col+other.col)
    def __sub__(self, other: "Pixel") -> "Pixel":
        return Pixel(row=self.row-other.row, col=self.col-other.col)
    def __isub__(self, other: "Pixel") -> "Pixel":
        return Pixel(row=self.row-other.row, col=self.col-other.col)
    def __neg__(self) -> "Pixel":
        return Pixel(row=-self.row, col=-self.col)
    def __pos__(self) -> "Pixel":
        return self

