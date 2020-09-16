Glossary
========

.. glossary::

It's not always easy to understand all the notions used in ``pyRDF2Vec``. This
glossary is here to help you to have an idea behind all these notions:

Continuous Bag-of-Words (CBOW)
   Model, part of ``Word2vec``, that predicts target words from contextual words
   in a given window.

depth
   Refers to the number of hops needed to reach sub-trees.

embedding technique
   Technique used in machine learning to represent complex objects (*e.g.,*
   texts, images, graphs) into a vector with a reduced number of features
   compared to the dimension of the dataset, while keeping the most important
   information about them.

embedding (or vector)
   Numerical representation of a word, regardless of where the words occurs in
   a sentence.

entity
   Specific type of node in a Knowledge Graph that is characterized by a URI.

entities
   Sequences that allows unsupervised feature extraction using language
   modeling.

feature matrix
   2D vector created from a Knowledge Graph for downstream Machine
   Learning (ML) tasks.

Knowledge Graph (KG)
   Initiated by Google, it unify information from various sources and enrich
   classical data formats by explicitly encoding relations between different data
   points in the form of edges.

Knowledge Graph Embeddings (KGE)
   Contains a set of entities and relationships between these entities, where
   all the facts in a Knowledge Graph are represented as triples (subject,
   predicate, object).

object
   Noun or pronoun used in a sentence and which is acted upon by the subject.


predicate
   Often associated with a large part of a sentence, designating something from
   the subject concerned in that sentence.

RDF2Vec
   Unsupervised technique that can create task-agnostic numerical
   representations of the nodes in a Knowledge Graph by extending successful
   language modeling techniques.

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
    Transform the Knowledge Graph into sequences of entities, which can be considered as sentences.

Uniform Resource Identifier (URI)
   Unique character string that identifies a particular resource, using a
   predefined set of syntax rules.

walking strategy
   Generates graph walks for each vertex of a given knowledge graph, from a
   certain depth.

Word2vec
   Neural language modeling techniques (NLP), which takes sequences of words to
   embed words into vector spaces.
