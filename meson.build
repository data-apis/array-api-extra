project(
  'array-api-extra',
  version: '0.10.0'
)

py = import('python').find_installation()

# NOTE: downstream projects may rely on this variable name when vendoring,
# do not change it without consulting with downstream projects
sources = {
  'array_api_extra': files(
    'src/array_api_extra/__init__.py',
    'src/array_api_extra/_delegation.py',
    'src/array_api_extra/testing.py',
  ),
  'array_api_extra/_lib': files(
    'src/array_api_extra/_lib/__init__.py',
    'src/array_api_extra/_lib/_at.py',
    'src/array_api_extra/_lib/_backends.py',
    'src/array_api_extra/_lib/_funcs.py',
    'src/array_api_extra/_lib/_lazy.py',
    'src/array_api_extra/_lib/_testing.py',
  ),
  'array_api_extra/_lib/_utils': files(
    'src/array_api_extra/_lib/_utils/__init__.py',
    'src/array_api_extra/_lib/_utils/_compat.py',
    'src/array_api_extra/_lib/_utils/_compat.pyi',
    'src/array_api_extra/_lib/_utils/_helpers.py',
    'src/array_api_extra/_lib/_utils/_typing.py',
    'src/array_api_extra/_lib/_utils/_typing.pyi',
  ),
}

foreach subdir, files : sources
  py.install_sources(files, subdir: subdir)
endforeach

subdir('tests')
subdir('vendor_tests')
