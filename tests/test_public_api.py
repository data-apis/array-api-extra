import inspect

from array_api_extra import (
    _agnostic,
    _at,
    _creation,
    _elementwise,
    _indexing,
    _lazy,
    _linalg,
    _manipulation,
    _searching,
    _set,
    _sorting,
    _statistical,
    testing,
)


def test_all_contains_all_public_functions():
    for module in (
        _at,
        _creation,
        _elementwise,
        _indexing,
        _lazy,
        _linalg,
        _manipulation,
        _searching,
        _set,
        _sorting,
        _statistical,
        _agnostic._creation,
        _agnostic._elementwise,
        _agnostic._indexing,
        _agnostic._inspection,
        _agnostic._linalg,
        _agnostic._manipulation,
        _agnostic._searching,
        _agnostic._set,
        _agnostic._sorting,
        _agnostic._statistical,
        testing._testing,
    ):

        def is_function_or_class(member: object):
            return inspect.isfunction(member) or inspect.isclass(member)

        public_functions_classes = {
            name
            for name, obj in inspect.getmembers(module, is_function_or_class)
            if not name.startswith("_") and obj.__module__ == module.__name__
        }
        missing = sorted(public_functions_classes - set(module.__all__))
        extra = sorted(set(module.__all__) - public_functions_classes)
        assert public_functions_classes == set(module.__all__), (
            f"{module.__name__}: Missing from __all__: {missing}\t"
            f"Extra in __all__: {extra}"
        )
