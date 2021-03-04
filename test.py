from pyrdf2vec.graphs import KG

KG(
    "http://10.2.35.70:5820/mutag",
    skip_predicates={"http://dl-learner.org/carcinogenesis#isMutagenic"},
),
