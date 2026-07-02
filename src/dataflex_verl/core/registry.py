"""Central registry for scorers and actuators.

Extends the original DataFlex Registry (DataFlex/src/dataflex/core/registry.py)
with a ``scorer`` kind and renames ``weighter`` -> ``reweighter`` to match the
Scorer/Actuator design. The build() semantics are preserved: cfg is merged with
runtime deps (runtime wins), then filtered to the callable's accepted kwargs so a
component only ever receives parameters it declares.
"""

import inspect
from typing import Any, Dict, Optional, Type


class Registry:
    def __init__(self):
        self._store: Dict[str, Dict[str, Type]] = {}

    def register(self, kind: str, name: str):
        def deco(cls: Type):
            self._store.setdefault(kind, {})
            if name in self._store[kind]:
                raise ValueError(f"{kind}.{name} already registered")
            self._store[kind][name] = cls
            return cls

        return deco

    def get(self, kind: str, name: str) -> Type:
        try:
            return self._store[kind][name]
        except KeyError:
            available = ", ".join(sorted(self._store.get(kind, {}))) or "<none>"
            raise KeyError(f"{kind}.{name} not registered. Available {kind}s: {available}")

    def list(self, kind: str):
        return sorted(self._store.get(kind, {}))

    def build(self, kind: str, name: str, *, runtime: Dict[str, Any], cfg: Optional[Dict[str, Any]] = None):
        cls = self.get(kind, name)
        cfg = cfg or {}
        merged = {**cfg, **runtime}  # runtime deps win over static cfg
        sig = inspect.signature(cls.__init__)
        params = list(sig.parameters.values())[1:]  # skip self
        accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)
        if accepts_kwargs:
            filtered = merged
        else:
            accepted = {p.name for p in params}
            filtered = {k: v for k, v in merged.items() if k in accepted}
        return cls(**filtered)


REGISTRY = Registry()


def register_scorer(name: str):
    return REGISTRY.register("scorer", name)


def register_selector(name: str):
    return REGISTRY.register("selector", name)


def register_reweighter(name: str):
    return REGISTRY.register("reweighter", name)


def register_mixer(name: str):
    return REGISTRY.register("mixer", name)
