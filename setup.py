"""Build the LF-SGG branched instance matcher (Cython/C++20) as a top-level extension.

The extension is named ``branched_ssg_matcher`` (matching the upstream module name used in the
.pyx/.pxd) so eval code can ``from branched_ssg_matcher import PyBranchedSSGMatcher``.

Requires a C++20 compiler (gcc>=11 / clang). Sources live in patchsgg/eval/branched/.
On non-POSIX platforms or when Cython is absent, the build degrades gracefully (the extension is
skipped) so pure-Python use still installs; eval that needs the matcher will then raise a clear
error at runtime.
"""
import sys

from setuptools import setup

ext_modules = []
try:
    from Cython.Build import cythonize
    from setuptools.extension import Extension

    if sys.platform == "win32":
        compile_args = ["/O2", "/std:c++20"]
    else:
        compile_args = ["-O3", "-std=c++20"]

    ext_modules = cythonize(
        [
            Extension(
                "branched_ssg_matcher",
                ["patchsgg/eval/branched/branched_ssg_matcher.pyx"],
                language="c++",
                extra_compile_args=compile_args,
            )
        ],
        language_level=3,
    )
except Exception as exc:  # pragma: no cover
    print(f"[patchsgg] skipping branched_ssg_matcher build ({exc}); install Cython + a C++20 compiler.")

setup(ext_modules=ext_modules)
