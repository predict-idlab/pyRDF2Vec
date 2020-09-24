"""Configuration file for the Sphinx documentation builder.

https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

import ast
import re
from pathlib import Path

import tomlkit

root = Path(__file__).parent.parent.absolute()
toml = tomlkit.loads((root / "pyproject.toml").read_text(encoding="utf8"))


def find(key: str) -> str:
    """Finds a value defined in the tool.poetry section of the pyproject.toml

    Args:
        key: The TOML key

    Returns:
        The TOML key's value

    """
    return str(toml["tool"]["poetry"][key])


author = re.sub(r"\s\<.+?\>", "", ", ".join(ast.literal_eval(find("authors"))))
copyright = "2020, " + find("license")
master_doc = "index"
project = find("name")
source_suffix = [".rst"]
version = find("version")

exclude_patterns = ["_build"]
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx_rtd_theme",
    "sphinxcontrib.apidoc",
]
intersphinx_mapping = {
    "rdflib": ("http://rdflib.readthedocs.org/en/latest/", None),
}
nitpick_ignore = [
    ("py:data", "typing.Any"),
    ("py:data", "typing.Dict"),
    ("py:data", "typing.Iterable"),
    ("py:data", "typing.List"),
    ("py:data", "typing.Optional"),
    ("py:data", "typing.Sequence"),
    ("py:data", "typing.Tuple"),
    ("py:data", "typing.Union"),
]
pygments_style = "sphinx"
templates_path = ["_templates"]

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "collapse_navigation": False,
    "display_version": True,
    "sticky_navigation": False,
}

apidoc_module_dir = "../pyrdf2vec"
apidoc_excluded_paths = ["tests"]
apidoc_separate_modules = True
