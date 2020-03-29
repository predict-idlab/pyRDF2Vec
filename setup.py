from setuptools import setup

setup(name='pyRDF2Vec',
      version='0.0.3',
      description='A python implementation of RDF2Vec',
      authors='Gilles Vandewiele, Bram Steenwinckel, Michael Weyns',
      author_email='gilles.vandewiele@ugent.be',
      url='https://github.com/IBCNServices/pyRDF2Vec',
      packages=['rdf2vec'],
      install_requires=['gensim', 'matplotlib', 'networkx', 'numpy', 
          'pandas', 'rdflib', 'scikit_learn', 'scipy', 'tqdm'
      ]
)