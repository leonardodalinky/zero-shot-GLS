# Available at setup time due to pyproject.toml
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import find_packages, setup

__version__ = "0.0.1"

# The main interface is through Pybind11Extension.
# * You can add cxx_std = 11 / 14 / 17, and then build_ext can be removed.
# * You can set include_pybind11 = false to add the include directory yourself,
# say from a submodule.
#
# Note:
# Sort input source files if you glob sources to ensure bit - for - bit
# reproducible builds(https: // github.com/pybind/python_example/pull/53)

ext_modules = [
    Pybind11Extension(
        "zgls_utils._huffman",
        ["src/huffman.cpp"],
        include_dirs=["src"],
        # Example : passing in the version to the compiled code
        define_macros=[("VERSION_INFO", __version__)],
        # -o3 and -g0
        extra_compile_args=["-O3", "-g0"],
    ),
]

setup(
    name="zgls-utils",
    version=__version__,
    author="Leonard Lin",
    author_email="leonard.keilin@gmail.com",
    description="Utility for Zero-shot GLS",
    long_description="",
    # `packages` for python packages
    packages=find_packages(),
    ext_modules=ext_modules,
    # Currently, build_ext only provides an optional "highest supported C++
    # level" feature, but in the future it may provide more features.
    cmdclass={"build_ext": build_ext},
    install_requires=["numpy>=1.7.0"],
    zip_safe=False,
    python_requires=">=3.7",
)
