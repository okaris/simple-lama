from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#") and not line.startswith("--")]

setup(
    name="simple-lama",
    version="0.1.0",
    author="Omer Karisman",
    author_email="ok@okaris.com",
    description="Simple script for LaMa inpainting using Hugging Face Hub",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/okaris/simple-lama",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "simple-lama=simple_lama.simple_lama:main_cli",
        ],
    },
) 