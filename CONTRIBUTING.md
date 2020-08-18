# Contributing to pyRDF2Vec

Hi, thank you for considering to contribute to pyRDF2Vec! There are many different ways in which you can help us out:

## Questions or reporting bugs

Feel free to [open an issue](https://github.com/IBCNServices/pyRDF2Vec/issues/new) in case something is not working as expected, or if you have any questions. In order to help you out as good as possible, please try to describe the question/problem as precise as possible. Also try to provide a minimal working example, the operating system you are using and some sample of data in case that is possible.

## Improving/extending the code

We also welcome any kinds of contributions in the form of pull requests: improvements to the documentation, optimizations of the code or new features that are related to [RDF2Vec](http://rdf2vec.org/). In case of larger changes, feel free to first open an issue to discuss this first. In order to keep the code consistent, we use the following tools

### [editorconfig](https://editorconfig.org/)

editorconfig keeps the code style consistent. An .editorconfig file is present in the repository. Some IDEs automatically support this, but others require you to install a plugin first.

### [pre-commit](https://github.com/pre-commit/pre-commit)

pre-commit performs all kinds of checks before a commit is performed, [it needs to be installed first as well](https://ljvmiranda921.github.io/notebook/2018/06/21/precommits-using-black-and-flake8/).

## Adding new walking strategies

As an extension to the original RDF2Vec algorithm, different walking strategies are supported in this project. In order to add your own strategy, you will have to extend the [Walker](https://github.com/IBCNServices/pyRDF2Vec/blob/master/rdf2vec/walkers/walker.py) interface. Your walker strategy should thus at least implement one method: `extract(graph, instances)`. In this method, the input is a [KnowledgeGraph](https://github.com/IBCNServices/pyRDF2Vec/blob/master/rdf2vec/graph.py) object and a list of identifiers that we can link to nodes from which we want to extract walks from. Many different examples of custom walking strategies can be found in [this directory](https://github.com/IBCNServices/pyRDF2Vec/tree/master/rdf2vec/walkers).