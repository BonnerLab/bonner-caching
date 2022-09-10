from typing import Any, ParamSpec, TypeVar
from collections.abc import Callable, Mapping

from functools import wraps
from tqdm.dask import TqdmCallback
import inspect
from pathlib import Path
import os
import pickle

import numpy as np
import xarray as xr

P = ParamSpec("P")
R = TypeVar("R")

DEFAULT_PATH = Path(
    os.getenv("BONNER_CACHING_CACHE", str(Path.home() / ".cache" / "bonner-caching"))
)
DEFAULT_MODE = os.getenv("BONNER_CACHING_MODE", "normal")

MODES = {"normal", "readonly", "overwrite", "delete", "ignore"}


# TODO factorize the loaders and savers for different formats into their own classes; see bonner-brainio's Uploader/Downloader classes for inspiration


class Cacher:
    def __init__(
        self,
        *,
        path: Path = DEFAULT_PATH,
        mode: str = DEFAULT_MODE,
        identifier: str = None,
        filetype: str = "pickle",
        kwargs_save: Mapping[str, Any] = {},
        kwargs_load: Mapping[str, Any] = {},
    ) -> None:
        self.path = path
        self.path.mkdir(parents=True, exist_ok=True)

        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.mode = mode
        self.identifier = identifier
        self.filetype = filetype
        self.kwargs_save = kwargs_save
        self.kwargs_load = kwargs_load

    def __call__(
        self,
        function: Callable[P, R],
    ) -> Callable[P, R]:
        @wraps(function)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            call_args = self.get_args(function, *args, **kwargs)

            identifier = self.identifier.format(**call_args)

            if self.mode == "normal":
                if self.is_stored(identifier):
                    result = self.load(identifier, **self.kwargs_load)
                else:
                    result = function(*args, **kwargs)
                    self.save(result, identifier, **self.kwargs_save)
            elif self.mode == "readonly":
                if self.is_stored(identifier):
                    result = self.load(identifier, **self.kwargs_load)
                else:
                    result = function(*args, **kwargs)
            elif self.mode == "overwrite":
                result = function(*args, **kwargs)
                self.save(result, identifier, **self.kwargs_save)
            elif self.mode == "delete":
                if self.is_stored(identifier):
                    self.delete(identifier)
                result = function(*args, **kwargs)
            elif self.mode == "ignore":
                result = function(*args, **kwargs)
            return result

        return wrapper

    def is_stored(self, identifier: str) -> Path | None:
        path = self.path / identifier
        if path.exists():
            return path
        else:
            return None

    def save(self, result: Any, identifier: str, **kwargs_save) -> None:  # type: ignore  # result can be Any
        if result is None:
            return
        path = self.path / identifier
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.filetype == "numpy":
            np.save(self.path / identifier, result, **kwargs_save)
        elif self.filetype == "netCDF4":
            with TqdmCallback(desc="dask", leave=False):
                result.to_netcdf(self.path / identifier, **kwargs_save)
        elif self.filetype == "pickle":
            with open(self.path / identifier, "wb") as f:
                pickle.dump(result, f, **kwargs_save)

    def load(self, identifier: str, **kwargs_load) -> Any:  # type: ignore  # file contents can be Any
        filepath = self.is_stored(identifier)
        if self.filetype == "numpy":
            return np.load(filepath, **kwargs_load)
        elif self.filetype == "netCDF4":
            try:
                return xr.open_dataarray(filepath, **kwargs_load)
            except Exception:
                return xr.open_dataset(filepath, **kwargs_load)
        elif self.filetype == "pickle":
            with open(filepath, "rb") as f:
                return pickle.load(f, **kwargs_load)

    def delete(self, identifier: str) -> None:
        filepath = self.is_stored(identifier)
        filepath.unlink()

    def get_args(  # type: ignore  # arguments can be Any
        self, function: Callable[P, R], *args: P.args, **kwargs: P.kwargs
    ) -> dict[str, Any]:
        signature = inspect.signature(function)
        bound_arguments = signature.bind(*args, **kwargs)
        bound_arguments.apply_defaults()
        return bound_arguments.arguments
