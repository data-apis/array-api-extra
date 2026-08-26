project(
  'array-api-extra',
  version: '0.11.2'
)

py = import('python').find_installation()

# NOTE: downstream projects may rely on this variable name when vendoring,
# do not change it without consulting with downstream projects
sources = {
  'array_api_extra': files(
    'src/array_api_extra/__init__.py',
    'src/array_api_extra/_at.py',
    'src/array_api_extra/_creation.py',
    'src/array_api_extra/_elementwise.py',
    'src/array_api_extra/_indexing.py',
    'src/array_api_extra/_lazy.py',
    'src/array_api_extra/_linalg.py',
    'src/array_api_extra/_manipulation.py',
    'src/array_api_extra/_searching.py',
    'src/array_api_extra/_set.py',
    'src/array_api_extra/_sorting.py',
    'src/array_api_extra/_statistical.py',
    'src/array_api_extra/py.typed',
  ),
  'array_api_extra/_agnostic': files(
    'src/array_api_extra/_agnostic/__init__.py',
    'src/array_api_extra/_agnostic/_creation.py',
    'src/array_api_extra/_agnostic/_elementwise.py',
    'src/array_api_extra/_agnostic/_indexing.py',
    'src/array_api_extra/_agnostic/_inspection.py',
    'src/array_api_extra/_agnostic/_linalg.py',
    'src/array_api_extra/_agnostic/_manipulation.py',
    'src/array_api_extra/_agnostic/_searching.py',
    'src/array_api_extra/_agnostic/_set.py',
    'src/array_api_extra/_agnostic/_sorting.py',
    'src/array_api_extra/_agnostic/_statistical.py',
  ),
  'array_api_extra/testing': files(
    'src/array_api_extra/testing/__init__.py',
    'src/array_api_extra/testing/_testing.py',
  ),
  'array_api_extra/_lib': files(
    'src/array_api_extra/_lib/__init__.py',
    'src/array_api_extra/_lib/_backends.py',
    'src/array_api_extra/_lib/_compat.py',
    'src/array_api_extra/_lib/_compat.pyi',
    'src/array_api_extra/_lib/_helpers.py',
    'src/array_api_extra/_lib/_testing.py',
    'src/array_api_extra/_lib/_typing.py',
    'src/array_api_extra/_lib/_typing.pyi',
  ),
}

foreach subdir, files : sources
  py.install_sources(files, subdir: subdir)
endforeach

subdir('tests')
