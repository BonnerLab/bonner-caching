from typing import Any, Callable, ParamSpec, TypeVar

from functools import wraps
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


class Cacher:
    def __init__(
        self,
        *,
        path: Path = DEFAULT_PATH,
        mode: str = DEFAULT_MODE,
        identifier: str = None,
        filetype: str = "pickle",
    ) -> None:
        self.path = path
        self.path.mkdir(parents=True, exist_ok=True)

        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.mode = mode
        self.identifier = identifier
        self.filetype = filetype

    def __call__(self, function: Callable[P, R]) -> Callable[P, R]:
        @wraps(function)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            call_args = self.get_args(function, *args, **kwargs)

            identifier = self.identifier.format(**call_args)

            if self.mode == "normal":
                if self.is_stored(identifier):
                    result = self.load(identifier)
                else:
                    result = function(*args, **kwargs)
                    self.save(result, identifier)
            elif self.mode == "readonly":
                if self.is_stored(identifier):
                    result = self.load(identifier)
                else:
                    result = function(*args, **kwargs)
            elif self.mode == "overwrite":
                result = function(*args, **kwargs)
                self.save(result, identifier)
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

    def save(self, result: Any, identifier: str) -> None:  # type: ignore  # result can be Any
        if result is None:
            return
        path = self.path / identifier
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.filetype == "numpy":
            np.save(self.path / identifier, result)
        elif self.filetype == "netCDF4":
            result.to_netcdf(self.path / identifier, compute=True)
        elif self.filetype == "pickle":
            with open(self.path / identifier, "wb") as f:
                pickle.dump(result, f)

    def load(self, identifier: str) -> Any:  # type: ignore  # file contents can be Any
        filepath = self.is_stored(identifier)
        if self.filetype == "numpy":
            return np.load(filepath)
        elif self.filetype == "netCDF4":
            try:
                return xr.open_dataarray(filepath)
            except Exception:
                return xr.open_dataset(filepath)
        elif self.filetype == "pickle":
            with open(filepath, "rb") as f:
                return pickle.load(f)

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
