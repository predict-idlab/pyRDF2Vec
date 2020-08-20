import pypandoc
from setuptools import find_packages, setup

try:
    long_description = pypandoc.convert("README.md", "rst")
except (OSError, ImportError):
    long_description = open("README.md").read()

setup(
    name="pyRDF2Vec",
    version="0.0.5",
    authors="Gilles Vandewiele, Bram Steenwinckel, Michael Weyns",
    long_description=long_description,
    description="Python implementation and extension of RDF2Vec",
    author_email="gilles.vandewiele@ugent.be",
    url="https://github.com/IBCNServices/pyRDF2Vec",
    packages=find_packages(),
    keywords="embeddings knowledge-graph rdf2vec word2vec",
    install_requires=[
        "gensim",
        "matplotlib",
        "networkx",
        "numpy",
        "pandas",
        "python-louvain",
        "rdflib",
        "scikit_learn",
        "tqdm",
    ],
    project_urls={
        "Source": "https://github.com/IBCNServices/pyRDF2Vec",
        "Tracker": "https://github.com/IBCNServices/pyRDF2Vec/issues",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Software Development :: Build Tools",
    ],
)
