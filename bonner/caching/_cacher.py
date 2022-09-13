__all__ = ["Cacher"]

from typing import Any, ParamSpec, TypeVar
from collections.abc import Callable, Mapping
from functools import wraps
import inspect
from pathlib import Path
import os

from ._handlers import get_handler

P = ParamSpec("P")
R = TypeVar("R")

DEFAULT_PATH = Path(
    os.getenv("BONNER_CACHING_CACHE", str(Path.home() / ".cache" / "bonner-caching"))
)
DEFAULT_MODE = os.getenv("BONNER_CACHING_MODE", "normal")

MODES = {"normal", "readonly", "overwrite", "delete", "ignore"}


class Cacher:
    def __init__(  # type: ignore  # kwargs can be Any
        self,
        *,
        path: Path = DEFAULT_PATH,
        mode: str = DEFAULT_MODE,
        identifier: str = None,
        filetype: str = "pickle",
        kwargs_save: Mapping[str, Any] = {},
        kwargs_load: Mapping[str, Any] = {},
    ) -> Callable[[Callable[P, R]], Callable[P, R]]:
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
        func: Callable[P, R],
    ) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            call_args = self._get_args(func, *args, **kwargs)

            identifier = self.identifier.format(**call_args)

            if self.mode == "normal":
                if self._get_path(identifier):
                    result = self.load(identifier, **self.kwargs_load)
                else:
                    result = func(*args, **kwargs)
                    self.save(result, identifier, **self.kwargs_save)
            elif self.mode == "readonly":
                if self._get_path(identifier):
                    result = self.load(identifier, **self.kwargs_load)
                else:
                    result = func(*args, **kwargs)
            elif self.mode == "overwrite":
                result = func(*args, **kwargs)
                self.save(result, identifier, **self.kwargs_save)
            elif self.mode == "delete":
                if self._get_path(identifier):
                    self.delete(identifier)
                result = func(*args, **kwargs)
            elif self.mode == "ignore":
                result = func(*args, **kwargs)
            return result

        return wrapper

    def save(self, result: Any, identifier: str) -> None:  # type: ignore  # result can be Any
        path = self.path / identifier
        path.parent.mkdir(parents=True, exist_ok=True)

        handler = get_handler(filetype=self.filetype)
        handler.save(result=result, path=path, **self.kwargs_save)

    def load(self, identifier: str) -> Any:  # type: ignore  # file contents can be Any
        path = self._get_path(identifier)
        handler = get_handler(filetype=self.filetype)
        return handler.load(path=path, **self.kwargs_load)

    def delete(self, identifier: str) -> None:
        filepath = self._get_path(identifier)
        filepath.unlink()

    def _get_path(self, identifier: str) -> Path | None:
        path = self.path / identifier
        if path.exists():
            return path
        else:
            return None

    def _get_args(  # type: ignore  # arguments can be Any
        self, function: Callable[P, R], *args: P.args, **kwargs: P.kwargs
    ) -> dict[str, Any]:
        signature = inspect.signature(function)
        bound_arguments = signature.bind(*args, **kwargs)
        bound_arguments.apply_defaults()
        return bound_arguments.arguments
