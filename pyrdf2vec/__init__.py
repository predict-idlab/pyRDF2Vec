"""isort:skip_file"""
from .rdf2vec import RDF2VecTransformer
from IPython import get_ipython


def is_notebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except:
        return False  # Probably standard Python interpreter


if is_notebook:
    import nest_asyncio

    nest_asyncio.apply()

__all__ = [
    "RDF2VecTransformer",
]
__version__ = "0.1.1"
