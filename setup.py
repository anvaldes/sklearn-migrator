import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='sklearn-migrator',  # Este será el nombre visible en PyPI
    version='0.11.0',
    author="Alberto Valdés",
    author_email="alberto.valdes.gonzalez.96.2@gmail.com",
    description="A utility to migrate scikit-learn models between versions.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/anvaldes/sklearn-migrator",
    project_urls={
        "Documentation": "https://github.com/anvaldes/sklearn-migrator#readme",
        "Source": "https://github.com/anvaldes/sklearn-migrator"
    },
    packages=setuptools.find_packages(),
    install_requires=[
        'scikit-learn>=0.21.3',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
