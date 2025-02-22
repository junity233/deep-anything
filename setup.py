from setuptools import setup, find_packages
from deepanything.metadatas import VERSION

with open("README.md",encoding='utf-8') as f:
    long_description = f.read()

with open("requirements.txt") as f:
    requirements = f.readlines()

setup(
    name="deepanything",
    version=VERSION,
    author="Junity",
    author_email="1727636624@qq.com",
    description="DeepAnything is a project that provides DeepSeek R1's deep thinking capabilities for various large language models (LLMs).",
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "deepanything=deepanything.__main__:main",
        ],
    },
    classifiers=[
      "Programming Language :: Python :: 3",
      "License :: OSI Approved :: MIT License",
      "Operating System :: OS Independent",
    ],
    install_requires=requirements
)