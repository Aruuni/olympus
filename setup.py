from setuptools import Extension, setup

setup(
    name="tcp_sockopt",
    version="0.1.0",
    ext_modules=[
        Extension(
            "tcp_sockopt",
            sources=["tcp_sockopt.c"],
            extra_compile_args=["-O2", "-Wall", "-Wextra"],
        )
    ],
)
