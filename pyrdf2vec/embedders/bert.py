from typing import List

import numpy as np
import rdflib
import torch
import transformers
from sklearn.utils.validation import check_is_fitted

from pyrdf2vec.embedders import Embedder


class BERT(Embedder):
    """Defines BERT embedding technique."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, corpus: List[List[str]]) -> "BERT":
        """Fits the BERT model based on provided corpus.

        Args:
            corpus: The corpus.

        Returns:
            The fitted BERT model.

        """
        tokenizer = transformers.BertTokenizer.from_pretrained(
            "bert-base-uncased"
        )

        #
        # with encode:
        #
        # Token IDs: tensor([101, 100, 100, 100, 100, 100, 102,   0,   0,   0,   0,   0,   0,   0,
        #   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        #   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        #   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        #   0,   0,   0,   0,   0,   0,   0,   0])

        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.

        for walk in corpus:
            encoded_dict = tokenizer.encode_plus(
                walk,  # Sentence to encode.
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                max_length=16,  # Pad & truncate all sentences.
                pad_to_max_length=True,
                return_attention_mask=True,  # Construct attn. masks.
                return_tensors="pt",  # Return pytorch tensors.
            )
        # {'input_ids': tensor([[101, 100, 100, 100, 100, 100, 102,   0,   0,   0,   0,   0,   0,   0,
        #    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        #    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        #    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        #    0,   0,   0,   0,   0,   0,   0,   0]]),

        # 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        # 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        # 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),

        # 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        # 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        # 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])}

        model = transformers.BertModel.from_pretrained(
            "bert-base-uncased",
            output_hidden_states=True,
        )
        # Put the model in "evaluation" mode, meaning feed-forward operation.
        model.eval()
        with torch.no_grad():
            self.model_ = model(
                input_ids=encoded_dict["input_ids"],
                attention_mask=encoded_dict["attention_mask"],
            )
        return self

    def transform(
        self, entities: List[rdflib.URIRef], verbose=True
    ) -> List[str]:
        """Constructs a features vector for the provided entities.

        Args:
            entities: The entities to create the embeddings.
                The test entities should be passed to the fit method as well.

                Due to RDF2Vec being unsupervised, there is no label leakage.

        Returns:
            The embeddings of the provided entities.

        """
        check_is_fitted(self, ["model_"])
        hidden_states = self.model_[2]
        if verbose:
            print(f"Layers: {len(hidden_states)}")
            print(f"Batches: {len(hidden_states[0])}")
            print(f"Tokens: {len(hidden_states[0][0])}")
            print(f"Hidden units: {len(hidden_states[0][0][0])}")

        # Concatenate the tensors for all layers as the final goal is to return
        # a single list of embeddings??? I am not sure about it.
        token_embeddings = torch.stack(hidden_states, dim=0)

        # The batch size, is used when submitting multiple sentences to the
        # model at once; here, though, we just have one example sentence.
        #
        # Let's get rid of the “batches” dimension since we don’t need it.
        token_embeddings = torch.squeeze(token_embeddings, dim=1)

        # Switch around the “layers” and “tokens” dimensions. I am not sure why
        # we need to do that.
        token_embeddings = token_embeddings.permute(1, 0, 2)

        # To create walk vectors, we summing the vectors from the last four
        # layers and add them to a list.
        token_vecs_sum = [
            torch.sum(token[-4:], dim=0) for token in token_embeddings
        ]

        if verbose:
            print(f"Shape is: {len(token_vecs_sum)}x{len(token_vecs_sum[0])}")

        # To get a single vector for our entire Knowledge Graph, we have multiple
        # application-dependent strategies, but a simple approach is to average
        # the second to last hiden layer of each token producing a single 768
        # length vector.
        #
        # Calculate the average of all 16 token vectors.
        sentence_embedding = torch.mean(hidden_states[-2][0], dim=0)

        # Let's return a list of embeddings instead of a tensor.
        return sentence_embedding.numpy()
