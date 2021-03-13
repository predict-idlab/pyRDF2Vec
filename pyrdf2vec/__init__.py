from .rdf2vec import RDF2VecTransformer


def is_notebook() -> bool:
    """Checks if the environment corresponds to a Notebook.

    Returns:
        True if the environment is a Notebook, False otherwise.

    """
    try:
        from IPython import get_ipython

        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except Exception:
        return False  # Probably standard Python interpreter


if is_notebook:
    # bypass the asyncio.run error in the Notebooks.
    import nest_asyncio

    nest_asyncio.apply()

__all__ = [
    "RDF2VecTransformer",
]
__version__ = "0.1.1"
