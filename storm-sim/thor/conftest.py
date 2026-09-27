"""Isolate historical QA profiles that configure shared module builders in place."""
import sys
import pytest


@pytest.fixture(autouse=True)
def restore_qa_profile_globals():
    modules = {name: (module, dict(vars(module))) for name, module in list(sys.modules.items())
               if name.startswith('tools.storm.benchgen.stream_eqa.')}
    yield
    for module, values in modules.values():
        for key in set(vars(module)) - set(values):
            delattr(module, key)
        vars(module).update(values)
