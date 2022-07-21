from collections.abc import Iterable
from typing import Any, Callable

from functools import wraps
import inspect
from pathlib import Path
import os
import pickle

import numpy as np
import xarray as xr


_BONNER_CACHING_HOME = Path(os.getenv("CACHE_HOME", str(Path.home() / "cache")))
_BONNER_CACHING_MODE = os.getenv("BONNER_CACHING_MODE", "normal")


class _Cacher:
    def __init__(
        self,
        *,
        include: Iterable[str] = [],
        exclude: Iterable[str] = [],
        custom_identifier: str = "",
    ) -> None:
        assert not (
            include and exclude
        ), "only one of 'include_args' and 'exclude_args' can be specified"
        self.cache_dir = _BONNER_CACHING_HOME
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.custom_identifier = custom_identifier
        self.include_args = include
        self.exclude_args = exclude

    def __call__(self, function: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            call_args = self.get_call_args(function, *args, **kwargs)
            identifier = self.create_identifier(function, call_args)

            if _BONNER_CACHING_MODE == "normal":
                if self.is_stored(identifier):
                    result = self.load(identifier)
                else:
                    result = function(*args, **kwargs)
                    self.save(result, identifier)
            elif _BONNER_CACHING_MODE == "readonly":
                if self.is_stored(identifier):
                    result = self.load(identifier)
                else:
                    result = function(*args, **kwargs)
            elif _BONNER_CACHING_MODE == "overwrite":
                result = function(*args, **kwargs)
                self.save(result, identifier)
            elif _BONNER_CACHING_MODE == "delete":
                if self.is_stored(identifier):
                    self.delete(identifier)
                result = function(*args, **kwargs)
            elif _BONNER_CACHING_MODE == "ignore":
                result = function(*args, **kwargs)
            else:
                raise ValueError(
                    f"$BONNER_CACHING_MODE cannot take the value {_BONNER_CACHING_MODE}"
                )
            return result

        return wrapper

    def is_stored(self, identifier: str) -> Path | None:
        filepaths = list(self.cache_dir.glob(identifier))
        assert len(filepaths) <= 1, "more than one file matches this identifier"
        if len(filepaths) == 1:
            return filepaths[0]
        else:
            return None

    def save(self, result: Any, identifier: str) -> None:
        filepath = self.cache_dir / identifier
        filepath.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(result, np.ndarray):
            np.save(self.cache_dir / f"{identifier}.npy", result)
        elif isinstance(result, xr.DataArray) or isinstance(result, xr.Dataset):
            result.to_netcdf(self.cache_dir / f"{identifier}.nc")
        else:
            with open(self.cache_dir / f"{identifier}.pkl", "wb") as f:
                pickle.dump(result, f)

    def load(self, identifier: str) -> Any:
        filepath = self.is_stored(identifier)
        if filepath.suffix == ".npy":
            return np.load(filepath)
        elif filepath.suffix == ".nc":
            return xr.load_dataarray(filepath)
        else:
            with open(filepath, "rb") as f:
                return pickle.load(f)

    def delete(self, identifier: str) -> Any:
        filepath = self.is_stored(identifier)
        filepath.unlink()

    def get_call_args(
        self, function: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> dict[str, Any]:
        signature = inspect.signature(function)
        bound_arguments = signature.bind(*args, **kwargs)
        bound_arguments.apply_defaults()
        return bound_arguments.arguments

    def create_identifier(
        self, function: Callable[..., Any], call_args: dict[str, Any]
    ) -> str:
        if self.exclude_args:
            call_args = {
                key: value
                for key, value in call_args.items()
                if key not in self.exclude_args
            }
        elif self.include_args:
            call_args = {
                key: value
                for key, value in call_args.items()
                if key in self.include_args
            }
        identifier = create_identifier(function, call_args)
        if self.custom_identifier:
            identifier = f"{identifier}_{self.custom_identifier}"
        return identifier


def create_identifier(function: Callable[..., Any], call_args: dict[str, Any]) -> str:
    module = [function.__module__, function.__name__]
    if "self" in call_args:
        object = call_args["self"]
        class_name = object.__class__.__name__
        if "object at" in str(object):
            object = class_name
        else:
            object = f"{class_name}({str(object)})"
        module.insert(1, object)
        del call_args["self"]
    module = ".".join(module)
    params = ",".join(
        f"{key}={str(value).replace('/', '_')}" for key, value in call_args.items()
    )
    if params:
        identifier = str(Path(module) / params)
    else:
        identifier = str(Path(module) / "_")
    return identifier