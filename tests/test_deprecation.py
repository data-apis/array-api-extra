from types import ModuleType

import pytest

from array_api_extra import broadcast_shapes, expand_dims


class TestDeprecatedFunctions:
    def test_broadcast_shapes(self, xp: ModuleType):
        with pytest.raises(DeprecationWarning, match=r"removed in v1.0.0"):
            _ = broadcast_shapes((2, 3), (2, 1), xp=xp)

    def test_expand_dims(self, xp: ModuleType):
        with pytest.raises(DeprecationWarning, match=r"removed in v1.0.0"):
            _ = expand_dims(xp.ones(2), axis=0, xp=xp)
