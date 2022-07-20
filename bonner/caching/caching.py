from typing import Any, Callable, Iterable
import numpy.typing as npt
from functools import wraps
import inspect
from pathlib import Path
import os

import numpy as np
import xarray as xr


DO_NOT_CACHE = True if os.getenv("DISK_CACHE_DO_NOT_CACHE", "") else False
USE_ONLY_CACHE = True if os.getenv("DISK_CACHE_USE_ONLY CACHE", "") else False
OVERWRITE_CACHE = True if os.getenv("DISK_CACHE_OVERWRITE_CACHE", "") else False


class Cacher:
    def __init__(
        self,
        *,
        include_args: Iterable[str],
        exclude_args: Iterable[str],
        custom_identifier: str,
        force_pickle: bool,
        cache_dir: Path,
    ) -> None:
        assert not (
            include_args and exclude_args
        ), "only one of 'include_args' and 'exclude_args' can be specified"
        if cache_dir:
            self.cache_dir = cache_dir
        else:
            self.cache_dir = Path(os.getenv("CACHE_HOME", str(Path.home() / "cache")))

    def __call__(self, function: Callable) -> Callable:
        @wraps
        def wrapper(*args, **kwargs):
            argspec = inspect.getfullargspec(function)
            called_args_ = 
        pass

    def save(self, result: Any, identifier: str) -> None:
        raise NotImplementedError()

    def load(self, identifier: str) -> None:
        raise NotImplementedError()

    def is_stored(self, identifier: str) -> bool:
        raise NotImplementedError()


class NumpyCacher(Cacher):
    def __init__(
        self,
        *,
        include_args: Iterable[str],
        exclude_args: Iterable[str],
        custom_identifier: str,
        force_pickle: bool = False,
    ) -> None:
        super().__init__(
            include_args=include_args,
            exclude_args=exclude_args,
            custom_identifier=custom_identifier,
            force_pickle=force_pickle,
        )

    def save(self, result: npt.NDArray, identifier: str):
        result.save()
        return super().save(result, identifier)


def _identify_backend(data: Any):
    if isinstance(data, np.ndarray):
        return "numpy"
    elif isinstance(data, xr.DataArray) or isinstance(data, xr.Dataset):
        return "xarray"
    else:
        return "pickle"
