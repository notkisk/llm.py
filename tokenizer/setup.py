from setuptools import setup, Extension

module = Extension(
    'tokenizer',
    sources=['tokenizer.c', 'tokenizer_python.c'],
    include_dirs=['.'],
    language='c',
)

setup(
    name='tokenizer',
    version='1.0.0',
    description='C-based BPE tokenizer with Python bindings',
    ext_modules=[module],
    py_modules=[],
)

