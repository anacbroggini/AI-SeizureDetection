import timeit

import numpy as np
import pandas as pd
# import modin.pandas as pd
from numpy.random import randn, randint
import pyarrow as pa
from pathlib import Path

def generate_data(n):
    df = pd.DataFrame(
        {
            "dt": randint(1_600_000_000, 1_700_000_000, size=n),
            "a": randn(n),
            "b": randn(n),
            "c": randn(n),
        },
        # dtype='double[pyarrow]' # doubles speed dor pyarrow reading  and improves reading feather/parquet. hdf won't work
        dtype=pd.ArrowDtype(pa.float64()) # doubles speed dor pyarrow reading  and improves reading feather/parquet. hdf won't work
    )
    df.dt = pd.to_datetime(df.dt, unit="s")
    df.set_index("dt", inplace=True)
    return df


def benchmark(df, name, saver, loader):
    verify(df, loader, saver)
    save_timer = timeit.Timer(lambda: saver(df))
    load_timer = timeit.Timer(lambda: loader().a.sum())
    save_n, save_time = save_timer.autorange()
    load_n, load_time = load_timer.autorange()
    total_time = (load_time / load_n) + (save_time / save_n)
    print(
        f"{name:<15s} : "
        f"{save_n / save_time:>20.3f} save/s : "
        f"{load_n / load_time:>20.3f} load+sum/s : "
        f"{1 / total_time: >20.3f} total speed"
    )


def verify(df, loader, saver):
    saver(df)
    loaded = loader()
    assert np.allclose(loaded.a.sum(), df.a.sum())
    assert np.allclose(loaded.b.sum(), df.b.sum())
    assert list(loaded.columns) == list(df.columns), loaded.columns


def save_feather(df):
    df = df.reset_index()
    df.to_feather("dummy.feather")


def load_feather():
    df = pd.read_feather("dummy.feather")
    df.set_index("dt", inplace=True)
    return df

def save_pyarrow(df):
    df = df.reset_index()
    arrow_filepath = Path('./data') / 'tmp_df'
    table = pa.Table.from_pandas(df, preserve_index=True)
    with pa.OSFile(str(arrow_filepath), 'wb') as sink:
        with pa.RecordBatchFileWriter(sink, table.schema) as writer:
            writer.write_table(table)


def load_pyarrow():
    arrow_filepath = Path('./data') / 'tmp_df'
    source = pa.memory_map(str(arrow_filepath), 'r')
    df = pa.ipc.RecordBatchFileReader(source).read_pandas()
    # df = pd.read_feather("dummy.feather")
    df.set_index("dt", inplace=True)
    return df


def main():
    df = generate_data(5_000_000)
    # benchmark(df, "dummy", lambda df: None, lambda: df)
    # benchmark(df, "csv", lambda df: df.to_csv("dummy.csv"), lambda: pd.read_csv("dummy.csv", index_col="dt"))
    benchmark(df, "pyarrow", save_pyarrow, load_pyarrow)
    # benchmark(df, "hdf", lambda df: df.to_hdf("dummy.h5", "dummy"), lambda: pd.read_hdf("dummy.h5", "dummy"))
    benchmark(df, "pickle", lambda df: df.to_pickle("dummy.pickle"), lambda: pd.read_pickle("dummy.pickle"))
    benchmark(df, "feather", save_feather, load_feather)
    benchmark(
        df,
        "parquet",
        lambda df: df.to_parquet("dummy.parquet", allow_truncated_timestamps=True),
        lambda: pd.read_parquet("dummy.parquet"),
    )


if __name__ == "__main__":
    main()