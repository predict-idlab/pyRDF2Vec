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
        str: The TOML key's value
    """
    return str(toml["tool"]["poetry"][key])


author = re.sub(r"\s\<.+?\>", "", ", ".join(ast.literal_eval(find("authors"))))
copyright = "2020, " + find("license")
master_doc = "index"
project = "pyRDF2Vec"
source_suffix = [".rst", ".md"]
version = find("version")

exclude_patterns = ["_build"]
extensions = ["sphinx_rtd_theme", "sphinxcontrib.napoleon"]
pygments_style = "sphinx"
templates_path = ["_templates"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
