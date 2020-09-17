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
   Refers to the number of hops needed to reach sub-trees.

community hops
   Extraction walk strategy that captures implicit relationships between nodes
   that are not explicitly modeled in the given Knowledge Graph.

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

entities
   Sequences that allows unsupervised feature extraction using language
   modeling.

feature matrix
   2D vector created from a Knowledge Graph for downstream Machine
   Learning (ML) tasks.

Hierarchical Random Walk (HALK)
   Transformation walk strategy that removes rare entities from random walks in
   order to increase the quality of generated embedding while decreasing memory
   usage.

Knowledge Graph (KG)
   Initiated by Google, it unify information from various sources and enrich
   classical data formats by explicitly encoding relations between different data
   points in the form of edges.

Knowledge Graph Embeddings (KGE)
   Contains a set of entities and relationships between these entities, where
   all the facts in a Knowledge Graph are represented as triples (subject,
   predicate, object).

N-Gram walks
   The transformation walk strategy based on the fact that the predecessors of
   a node that two different walks have in common can be different.

object
   Noun or pronoun used in a sentence and which is acted upon by the subject.

predicate
   Often associated with a large part of a sentence, designating something from
   the subject concerned in that sentence.

RDF2Vec
   Unsupervised technique that can create task-agnostic numerical
   representations of the nodes in a Knowledge Graph by extending successful
   language modeling techniques.

sampling
   Gets a collection of biased walks after each iteration, according to metric.

sampling strategy
   Select relevant neighbor entities.

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

subject
   Noun or pronoun used in a sentence and related to an action.

transformer
   Transform the Knowledge Graph into sequences of entities, which can be
   considered as sentences.

Uniform Resource Identifier (URI)
   Unique character string that identifies a particular resource, using a
   predefined set of syntax rules.

vertice
   Node which can be one of the three following types: entity, blank or literal.

walk
   Sequence of vertices that can be found in the Knowledge Graph by browsing
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
