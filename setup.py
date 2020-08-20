import pypandoc
from setuptools import find_packages, setup

try:
    long_description = pypandoc.convert("README.md", "rst")
except (OSError, ImportError):
    long_description = open("README.md").read()

setup(
    name="pyRDF2Vec",
    version="0.0.5",
    description="A python implementation of RDF2Vec",
    authors="Gilles Vandewiele, Bram Steenwinckel, Michael Weyns",
    long_description=long_description,
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
)
