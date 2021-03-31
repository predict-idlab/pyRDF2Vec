import itertools

import numpy as np
import pytest

from pyrdf2vec.embedders import Doc2Vec

CORPUS_LOOP = [
    [
        (
            "http://pyRDF2Vec#Alice",
            "http://pyRDF2Vec#knows",
            "b'\\x8b\\x93\\x8dx\\x1c\\xc7\\xd3\\xc9'",
            "http://pyRDF2Vec#loves",
            "b'\\x94\\xefX\\x0c\\x04l4E'",
        ),
        (
            "http://pyRDF2Vec#Bob",
            "http://pyRDF2Vec#knows",
            "b'\\x8b\\x93\\x8dx\\x1c\\xc7\\xd3\\xc9'",
            "http://pyRDF2Vec#loves",
            "b'\\x94\\xefX\\x0c\\x04l4E'",
        ),
        (
            "http://pyRDF2Vec#Dean",
            "http://pyRDF2Vec#loves",
            "b'\\x94\\xefX\\x0c\\x04l4E'",
            "http://pyRDF2Vec#knows",
            "b'\\x1eK\\xad\\xc6\\xb6\\x1cu:'",
        ),
        (
            "http://pyRDF2Vec#Dean",
            "http://pyRDF2Vec#loves",
            "b'\\x94\\xefX\\x0c\\x04l4E'",
            "http://pyRDF2Vec#knows",
            "b'\\x8b\\x93\\x8dx\\x1c\\xc7\\xd3\\xc9'",
        ),
        (
            "http://pyRDF2Vec#Alice",
            "http://pyRDF2Vec#knows",
            "b'\\x1eK\\xad\\xc6\\xb6\\x1cu:'",
            "http://pyRDF2Vec#knows",
            "b'\\x8b\\x93\\x8dx\\x1c\\xc7\\xd3\\xc9'",
        ),
    ]
]

CORPUS_CHAIN = [
    [
        ("http://pyRDF2Vec#Dean",),
        (
            "http://pyRDF2Vec#Alice",
            "http://pyRDF2Vec#knows",
            "b'\\x1eK\\xad\\xc6\\xb6\\x1cu:'",
            "http://pyRDF2Vec#knows",
            "b'1\\xa1\\x90\\xf2e\\x8a%\\x17'",
        ),
        (
            "http://pyRDF2Vec#Bob",
            "http://pyRDF2Vec#knows",
            "b'1\\xa1\\x90\\xf2e\\x8a%\\x17'",
            "http://pyRDF2Vec#knows",
            "b'\\x87)K\\xbf5!\\x13\\x00'",
        ),
        (
            "http://pyRDF2Vec#Alice",
            "http://pyRDF2Vec#knows",
            "b'\\x8b\\x93\\x8dx\\x1c\\xc7\\xd3\\xc9'",
        ),
    ]
]

CORPUS = [CORPUS_LOOP, CORPUS_CHAIN]
IS_UPDATE = [False, True]
ROOTS_WITHOUT_URL = ["Alice", "Bob", "Dean"]
URL = "http://pyRDF2Vec"


class TestDoc2Vec:
    @pytest.mark.parametrize("corpus", CORPUS)
    def test_determinism(self, corpus):
        w1 = (
            Doc2Vec(workers=1)
            .fit(corpus)
            .transform([f"{URL}#{entity}" for entity in ROOTS_WITHOUT_URL])
        )
        w2 = (
            Doc2Vec(workers=1)
            .fit(corpus)
            .transform([f"{URL}#{entity}" for entity in ROOTS_WITHOUT_URL])
        )
        assert np.array_equal(w1, w2)

    @pytest.mark.parametrize(
        "corpus, root",
        list(itertools.product(CORPUS, ROOTS_WITHOUT_URL)),
    )
    def test_fit(self, corpus, root):
        d2v = Doc2Vec()
        with pytest.raises(KeyError):
            d2v._model.wv[f"{URL}#{root}"]
        assert len(d2v._model.wv.key_to_index) == 0
        d2v.fit(corpus, False)
        if corpus == CORPUS_LOOP:
            assert len(d2v._model.wv.key_to_index) == 5
        else:
            assert len(d2v._model.wv.key_to_index) == 4

    @pytest.mark.parametrize("corpus", CORPUS)
    def test_transform(self, corpus):
        d2v = Doc2Vec()
        d2v.fit(corpus)
        embeddings = d2v.transform(
            [f"{URL}#{entity}" for entity in ROOTS_WITHOUT_URL]
        )
        assert len(embeddings) > 0

    @pytest.mark.parametrize("corpus", CORPUS)
    def test_online_training(self, corpus):
        d2v = Doc2Vec(workers=1)
        d2v.fit(corpus)
        embeddings = d2v.transform(
            [f"{URL}#{entity}" for entity in ROOTS_WITHOUT_URL]
        )
        assert len(embeddings) == 3
        corpus.append(
            [
                (
                    "http://pyRDF2Vec#Alice",
                    "http://pyRDF2Vec#knows",
                    "http://pyRDF2Vec#Casper",
                    "http://pyRDF2Vec#knows",
                    "http://pyRDF2Vec#Mario",
                )
            ]
        )
        d2v.fit(corpus, True)
        embeddings = d2v.transform(
            [f"{URL}#{entity}" for entity in ROOTS_WITHOUT_URL]
        )
        assert len(embeddings) == 3
