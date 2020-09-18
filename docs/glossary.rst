Glossary
========

.. glossary::

It's not always easy to understand all the notions used in ``pyRDF2Vec``. This
glossary is here to help you to have an idea behind all these notions:

anonymous walks
   Transformation walking strategy that transforms label information into
   positional information.

Continuous Bag-of-Words (CBOW)
   Model, part of ``Word2vec``, that predicts target words from contextual words
   in a given window.

depth
   Refers to the number of hops in a walk.

community hops
   A hop to a node that is not a neighbor, but is rather part of the same
   community, which is determined through community detection.

community walks
   An extraction walk strategy that allows for community hops with a certain
   probability.

embedding technique
   Technique used in machine learning to represent complex objects (*e.g.,*
   texts, images, graphs) into a vector with a reduced number of features
   compared to the dimension of the dataset, while keeping the most important
   information about them.

embeddings (or latent representation/vectors)
   Numerical representation of a node in a given Knowledge Graph, where
   entities that are semantically related should be close to each other in the
   embedded space.

entity
   Specific type of node in a Knowledge Graph that is characterized by a URI.

feature matrix
   An NxK matrix where N is the number of entities and K the embedding size,
   which can be used for further downstream Machine learning (ML) tasks.

Hierarchical Random Walks (HALK)
   Transformation walk strategy that removes rare entities from random walks.

Knowledge Graph (KG)
   A graphical representation of (domain or expert) knowledge encoded as a
   collection of triples having the form (subject, predicate, object).

N-Gram walks
   The transformation walk strategy based on that creates N-grams from N
   consecutive hops in a walk, which are then relabeled.

RDF2Vec
   Unsupervised technique that can create task-agnostic numerical
   representations of the nodes in a Knowledge Graph by extending successful
   language modeling techniques.

sampling strategy
   A strategy to select the next neighbor in a walk. This can either be at
   random or guided by some metric (biased walks).

Skip-Gram (SG)
   Model, part of ``Word2vec``, that predicts the context words from the target
   words in a given window.

SPARQL Query Language (SPARQL)
   Declarative Query Language (*e.g.,* SQL) for performing Data Manipulation
   and Data Definition operations on Data represented as a collection of RDF
   Language sentences/statements.

SPARQL endpoint
   Point of presence identified by a URL (SPARQL Endpoint URL) and located on
   an HTTP network that is capable of receiving and processing requests under
   the SPARQL protocol.

Uniform Resource Identifier (URI)
   Unique character string that identifies a particular resource, using a
   predefined set of syntax rules.

vertex
   Node in graph which can be one of the three following types: entity, blank
   or literal.

walk
   Sequence of vertices that can be found in the Knowledge Graph by traversing
   the given directed links.

walking strategy
   Generates graph walks for each vertex of a given knowledge graph, from a
   certain depth according to a type of a strategy (type 1 for extraction or
   type 2 for transformation).

walklets
   Transformation walking strategy with walks of length two, consisting of the
   root of the original walk and one of the hops.

Word2vec
   Neural language modeling techniques (NLP), which takes sequences of words to
   embed words into vector spaces.
