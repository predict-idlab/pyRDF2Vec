from typing import List

import numpy as np
import transformers
from sklearn.utils.validation import check_is_fitted
from torch.utils.data import Dataset
from transformers import (BertConfig, BertForMaskedLM, BertTokenizer,
                          DataCollatorForLanguageModeling, DistilBertConfig,
                          DistilBertForMaskedLM, DistilBertTokenizer, Trainer,
                          TrainingArguments)

from pyrdf2vec.embedders import Embedder


class WalkDataset(Dataset):
    def __init__(self, corpus, tokenizer):
        self.walks = [
            tokenizer(
                " ".join(walk), padding=True, truncation=True, max_length=512
            )
            for walk in corpus
        ]

    def __len__(self):
        return len(self.walks)

    def __getitem__(self, i):
        return self.walks[i]


class BERT(Embedder):
    """Defines BERT embedding technique."""

    def __init__(self, vocab_filename="bert-kg", seed=None):
        self.seed = seed
        self.vocab_filename = vocab_filename
        self.verbose = True
        self._vocabulary_size = 0

    def _build_vocabulary(
        self, hops: List[str], is_update: bool = False
    ) -> None:
        with open(self.vocab_filename, "w") as f:
            if not is_update:
                for token in ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]:
                    f.write("%s\n" % token)
                    self._vocabulary_size += 1
            for hop in hops:
                f.write("%s\n" % hop)
                self._vocabulary_size += 1

    def fit(self, corpus, is_update=False):
        unique_hops = list({hop for walk in corpus for hop in walk})
        self._build_vocabulary(unique_hops, is_update)

        # tokenizer = BertTokenizer(
        #     vocab_file=self.vocab_filename,
        #     do_lower_case=False,
        #     never_split=unique_hops,
        # )

        self.tokenizer = DistilBertTokenizer(
            vocab_file=self.vocab_filename,
            do_lower_case=False,
            never_split=unique_hops,
        )

        # self.model_ = BertForMaskedLM(
        #     BertConfig(
        #         vocab_size=self._vocabulary_size,
        #         max_position_embeddings=512,
        #         type_vocab_size=1,
        #     )
        # )

        self.model_ = DistilBertForMaskedLM(
            DistilBertConfig(
                vocab_size=self._vocabulary_size,
                max_position_embeddings=512,
                type_vocab_size=1,
            )
        )

        train_dataset = WalkDataset(corpus, self.tokenizer)

        print(
            self.model_.distilbert.embeddings.word_embeddings.weight[
                train_dataset[1]["input_ids"][1]
            ][:25]
        )

        Trainer(
            model=self.model_,
            args=TrainingArguments(
                output_dir="./bert",
                overwrite_output_dir=True,
                num_train_epochs=1,
                warmup_steps=500,
                weight_decay=0.2,
                logging_dir="./logs",
                seed=self.seed,
                dataloader_num_workers=2,
                prediction_loss_only=True,
            ),
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer
            ),
            train_dataset=train_dataset,
        ).train()

        print(
            self.model_.distilbert.embeddings.word_embeddings.weight[
                train_dataset[1]["input_ids"][1]
            ][:25]
        )
        return self

    def transform(self, entities: List[str]):
        check_is_fitted(self, ["model_"])
        return [
            self.model_.distilbert.embeddings.word_embeddings.weight[
                self.tokenizer(entity)["input_ids"][1]
            ]
            .cpu()
            .detach()
            .numpy()
            for entity in entities
        ]
