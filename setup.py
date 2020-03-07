from setuptools import setup

setup(name='pyRDF2Vec',
      version='0.0.2',
      description='A python implementation of RDF2Vec',
      author='Gilles Vandewiele',
      author_email='gilles.vandewiele@ugent.be',
      url='https://github.com/IBCNServices/pyRDF2Vec',
      packages=['rdf2vec'],
      install_requires=[
          'gensim==3.5.0',
          'matplotlib==2.1.1',
      	  'networkx==2.2',
      	  'numpy==1.13.3',
      	  'pandas==0.23.4',
      	  'rdflib==4.2.2',
      	  'scikit_learn==0.21.2',
      	  'scipy==0.19.1',
      	  'tqdm==4.19.5',
      ]
)