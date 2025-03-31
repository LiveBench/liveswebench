import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="liveswebench",
    author="Abacus.AI",
    description="LiveSWEBench is a benchmark for evaluating AI agents on software engineering tasks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
    install_requires=[
        'GitPython',
        'python-dotenv',
        'requests',
        'ghapi',
        'fastcore',
        'beautifulsoup4',
        'unidiff',
        'tqdm',
        'rich',
        'datasets',
        'chardet',
        'openai',
        'anthropic',
    ],
    extras_require={
        'base': ['GitPython'],
        'dev': ['pytest>=7.0.0', 'pytest-mock'],
    },
    include_package_data=True,
)