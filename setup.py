from setuptools import setup, find_packages

setup(
    name="deepanything",
    version="0.1.0",
    author="Junity",
    author_email="1727636624@qq.com",
    description="",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "deepanything=deepanything.__main__:main",
        ],
    },
)