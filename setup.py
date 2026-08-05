"""Build the LF-SGG branched instance matcher as a top-level extension."""
from pathlib import Path
import sys

from setuptools import setup


BRANCHED_DIR = Path("patchsgg/eval/branched")
ext_modules = []

try:
    from Cython.Build import cythonize
    from setuptools.extension import Extension

    compile_args = ["/O2", "/std:c++20"] if sys.platform == "win32" else ["-O3", "-std=c++20"]

    extension = Extension(
        "branched_ssg_matcher",
        [str(BRANCHED_DIR / "branched_ssg_matcher.pyx")],
        language="c++",
        extra_compile_args=compile_args,
        include_dirs=[str(BRANCHED_DIR)],
    )
    ext_modules = cythonize(
        [extension],
        language_level=3,
        include_path=[str(BRANCHED_DIR)],
    )
except Exception as exc:  # pragma: no cover
    print(
        "[patchsgg] skipping branched_ssg_matcher build "
        f"({exc}); install Cython and a C++20 compiler."
    )

setup(ext_modules=ext_modules)
