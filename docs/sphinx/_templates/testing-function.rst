{{ fullname.split('.')[-2:] | join('.') | escape | underline }}

.. currentmodule:: {{ module }}

.. autofunction:: {{ fullname }}