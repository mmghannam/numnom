"""Fast MPS file parser for mathematical programming."""

from ._numnom import (
    CONTINUOUS,
    INTEGER,
    SEMI_CONTINUOUS,
    SEMI_INTEGER,
    Model,
    SparseMatrix,
    parse_file,
    parse_str,
    write_file,
    write_str,
)


def to_scipy(model: "Model") -> "scipy.sparse.csc_matrix":  # noqa: F821
    """Convert a parsed Model's constraint matrix to a scipy.sparse.csc_matrix.

    Requires the optional ``scipy`` dependency::

        pip install "numnom[scipy]"
    """
    from scipy.sparse import csc_matrix

    a = model.a_matrix
    return csc_matrix(
        (a.value, a.index, a.start),
        shape=(model.num_row, model.num_col),
    )


__all__ = [
    "CONTINUOUS",
    "INTEGER",
    "SEMI_CONTINUOUS",
    "SEMI_INTEGER",
    "Model",
    "SparseMatrix",
    "parse_file",
    "parse_str",
    "to_scipy",
    "write_file",
    "write_str",
]
