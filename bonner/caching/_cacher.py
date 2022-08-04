from collections.abc import Iterable
from typing import Any, Callable, ParamSpec, TypeVar

from functools import wraps
import glob
import inspect
from pathlib import Path
import os
import pickle

import numpy as np
import xarray as xr

P = ParamSpec("P")
R = TypeVar("R")

DEFAULT_PATH = Path(
    os.getenv("BONNER_CACHING_PATH", str(Path.home() / ".cache" / "bonner-caching"))
)
DEFAULT_MODE = os.getenv("BONNER_CACHING_MODE", "normal")

MODES = {"normal", "readonly", "overwrite", "delete", "ignore"}


class Cacher:
    def __init__(
        self,
        *,
        path: Path = DEFAULT_PATH,
        mode: str = DEFAULT_MODE,
        include: Iterable[str] = [],
        custom_identifier: str = "",
    ) -> None:
        self.path = path
        self.path.mkdir(parents=True, exist_ok=True)

        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.mode = mode
        self.custom_identifier = custom_identifier
        self.include_args = include

    def __call__(self, function: Callable[P, R]) -> Callable[P, R]:
        # TODO add correct type annotations
        @wraps(function)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            call_args = self.get_args(function, *args, **kwargs)
            identifier = self.create_identifier(function, call_args)

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
        # TODO write more efficient version of this (e.g. filepaths = list((self.path / identifier).glob("*")))
        filepaths = list(self.path.rglob(f"{glob.escape(identifier)}*"))
        filepaths = [
            path
            for path in filepaths
            if str(path.relative_to(self.path).with_suffix("")) == identifier
        ]
        if len(filepaths) == 0:
            return None
        elif len(filepaths) == 1:
            return filepaths[0]
        else:
            raise ValueError(f"More than one file matches the identifier {identifier}")

    def save(self, result: Any, identifier: str) -> None:
        filepath = self.path / identifier
        filepath.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(result, np.ndarray):
            np.save(self.path / f"{identifier}.npy", result)
        elif isinstance(result, xr.DataArray):
            result.to_netcdf(self.path / f"{identifier}.nc")
        else:
            with open(self.path / f"{identifier}.pkl", "wb") as f:
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

    def delete(self, identifier: str) -> None:
        filepath = self.is_stored(identifier)
        filepath.unlink()

    def get_args(
        self, function: Callable[P, R], *args: P.args, **kwargs: P.kwargs
    ) -> dict[str, Any]:
        signature = inspect.signature(function)
        bound_arguments = signature.bind(*args, **kwargs)
        bound_arguments.apply_defaults()
        return bound_arguments.arguments

    def create_identifier(self, function: Callable[P, R], args: dict[str, Any]) -> str:
        if self.include_args:
            args = {
                key: value for key, value in args.items() if key in self.include_args
            }
        module_identifier, parameters_identifier = create_identifier(function, args)
        if parameters_identifier:
            if self.custom_identifier:
                filename = "_".join((parameters_identifier, self.custom_identifier))
            else:
                filename = parameters_identifier
        else:
            if self.custom_identifier:
                filename = self.custom_identifier
            else:
                raise ValueError(
                    "Custom identifier must be passed if no arguments are used for"
                    " naming"
                )

        return f"{module_identifier}/{filename}"


def create_identifier(
    function: Callable[P, R], args: dict[str, Any]
) -> tuple[str, str]:
    module = [function.__module__, function.__name__]
    if "self" in args:
        object = args["self"]
        class_name = object.__class__.__name__
        if "object at" in str(object):
            object = class_name
        else:
            object = f"{class_name}({str(object)})"
        module.insert(1, object)
        del args["self"]
    module_identifier = ".".join(module)
    parameters_identifier = ",".join(
        f"{key}={str(value).replace('/', '_')}" for key, value in args.items()
    )
    return module_identifier, parameters_identifier
