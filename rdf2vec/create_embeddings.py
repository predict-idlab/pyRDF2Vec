import random
import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import rdflib

from graph import rdflib_to_kg
from rdf2vec import RDF2VecTransformer

from walkers import (RandomWalker, WeisfeilerLehmanWalker, 
                     AnonymousWalker, WalkletWalker, NGramWalker,
                     CommunityWalker)

os.environ['PYTHONHASHSEED'] = '42'
random.seed(42)
np.random.seed(42)

strategies = [
	RandomWalker, 
    WeisfeilerLehmanWalker, 
    AnonymousWalker, 
    WalkletWalker, 
    NGramWalker,
    CommunityWalker
]

def measure(rdf_file, format, train_file, test_file, entity_col, label_col, label_predicates, output_dir, name):
    # Load in our graph using rdflib
    print(end='Loading data... ', flush=True)
    g = rdflib.Graph()
    if format is not None:
        g.parse(rdf_file, format=format)
    else:
        g.parse(rdf_file)
    print('OK')

    # Create some lists of train and test entities & labels
    test_data = pd.read_csv(train_file, sep='\t')
    train_data = pd.read_csv(test_file, sep='\t')

    train_entities = [rdflib.URIRef(x) for x in train_data[entity_col]]
    train_labels = train_data[label_col]

    test_entities = [rdflib.URIRef(x) for x in test_data[entity_col]]
    test_labels = test_data[label_col]

    all_entities = train_entities + test_entities
    all_labels = list(train_labels) + list(test_labels)

    # Convert the rdflib graph to our graph
    kg = rdflib_to_kg(g, label_predicates=label_predicates)

    results = {}
    results['ground_truth'] = test_labels

    for depth in [2, 4, 6, 8]:
        for sg in [0, 1]:
            for strategy in strategies:
                walker = strategy(depth, float('inf'))
                transformer = RDF2VecTransformer(walkers=[walker], sg=sg)
                embeddings = transformer.fit_transform(kg, all_entities)

                train_embeddings = embeddings[:len(train_entities)]
                test_embeddings = embeddings[len(train_entities):]

                results['train_embeddings'] = train_embeddings
                results['test_embeddings'] = test_embeddings

                pickle.dump(results, open('{}/{}_{}_{}_{}.p'.format(output_dir, name, depth, sg, strategy.__class__.__name__), 'wb+'))

for _ in range(10):

    ##################### MUTAG ####################################
    rdf_file = '../data/MUTAG/mutag.owl'
    format = None
    train_file = '../data/MUTAG/MUTAG_test.tsv'
    test_file = '../data/MUTAG/MUTAG_train.tsv'
    entity_col = 'bond'
    label_col = 'label_mutagenic'
    label_predicates = [
        rdflib.term.URIRef('http://dl-learner.org/carcinogenesis#isMutagenic')
    ]
    output = 'output/mutag.p'
    measure(rdf_file, format, train_file, test_file, entity_col, label_col, label_predicates, 'output', 'MUTAG')

    ###################### BGS #####################################
    rdf_file = '../data/BGS/BGS.nt'
    format = 'nt'
    train_file = '../data/BGS/BGS_test.tsv'
    test_file = '../data/BGS/BGS_train.tsv'
    entity_col = 'rock'
    label_col = 'label_lithogenesis'
    label_predicates = [
            rdflib.term.URIRef('http://data.bgs.ac.uk/ref/Lexicon/hasLithogenesis'),
            rdflib.term.URIRef('http://data.bgs.ac.uk/ref/Lexicon/hasLithogenesisDescription'),
            rdflib.term.URIRef('http://data.bgs.ac.uk/ref/Lexicon/hasTheme')
    ]
    output = 'output/bgs.p'
    measure(rdf_file, format, train_file, test_file, entity_col, label_col, label_predicates, 'output', 'BGS')

    ##################### AIFB #####################################
    rdf_file = '../data/AIFB/aifb.n3'
    format = 'n3'
    train_file = '../data/AIFB/AIFB_test.tsv'
    test_file = '../data/AIFB/AIFB_train.tsv'
    entity_col = 'person'
    label_col = 'label_affiliation'
    label_predicates = [
            rdflib.URIRef('http://swrc.ontoware.org/ontology#affiliation'),
            rdflib.URIRef('http://swrc.ontoware.org/ontology#employs'),
            rdflib.URIRef('http://swrc.ontoware.org/ontology#carriedOutBy')
    ]
    output = 'output/aifb.p'
    measure(rdf_file, format, train_file, test_file, entity_col, label_col, label_predicates, 'output', 'AIFB')

    ###################### AM ######################################
    rdf_file = '../data/AM/rdf_am-data.ttl'
    format = 'turtle'
    train_file = '../data/AM/AM_test.tsv'
    test_file = '../data/AM/AM_train.tsv'
    entity_col = 'proxy'
    label_col = 'label_cateogory'
    label_predicates = [
       rdflib.term.URIRef('http://purl.org/collections/nl/am/objectCategory'),
       rdflib.term.URIRef('http://purl.org/collections/nl/am/material')
    ]
    output = 'output/am.p'
    measure(rdf_file, format, train_file, test_file, entity_col, label_col, label_predicates, 'output', 'AM')


    ##################### CITESEER ####################################
    rdf_file = 'data/CITESEER/citeseer.ttl'
    label_file = 'data/CITESEER/label.txt'
    train_file = 'data/CITESEER/train.txt'
    test_file = 'data/CITESEER/test.txt'
    label_predicates = [
        rdflib.URIRef('http://hasLabel')
    ]
    output = 'output/citeseer.p'
    measure(rdf_file, format, train_file, test_file, entity_col, label_col, label_predicates, 'output', 'Citeseer')

    ##################### CORA ####################################
    rdf_file = 'data/CORA/cora.ttl'
    label_file = 'data/CORA/label.txt'
    train_file = 'data/CORA/train.txt'
    test_file = 'data/CORA/test.txt'
    val_file = 'data/CORA/dev.txt'
    label_predicates = [
        rdflib.URIRef('http://hasLabel')
    ]
    output = 'output/cora.p'
    measure(rdf_file, format, train_file, test_file, entity_col, label_col, label_predicates, 'output', 'Cora')

    ##################### PUBMED ####################################
    rdf_file = 'data/PUBMED/pubmed.ttl'
    label_file = 'data/PUBMED/label.txt'
    train_file = 'data/PUBMED/train.txt'
    test_file = 'data/PUBMED/test.txt'
    val_file = 'data/PUBMED/dev.txt'
    label_predicates = [
        rdflib.URIRef('http://hasLabel')
    ]
    output = 'output/pubmed.p'
    measure(rdf_file, format, train_file, test_file, entity_col, label_col, label_predicates, 'output', 'Pubmed')


