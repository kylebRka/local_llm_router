"""Compatibility command: implementation is llm_router.cli.router."""

import importlib as _importlib
import sys as _sys

_module = _importlib.import_module("llm_router.cli.router")
if __name__ == "__main__":
    _module.main()
else:
    _sys.modules[__name__] = _module
