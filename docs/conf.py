# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

author = "Gilles Vandewiele, Bram Steenwinckel, Michael Weyns"
copyright = "2020, Gilles Vandewiele, Bram Steenwinckel, Michael Weyns"
master_doc = "index"
project = "pyRDF2Vec"
source_suffix = [".rst", ".md"]
version = "0.5"

exclude_patterns = ["_build"]
extensions = ["sphinx_rtd_theme", "sphinxcontrib.napoleon"]
pygments_style = "sphinx"
templates_path = ["_templates"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
