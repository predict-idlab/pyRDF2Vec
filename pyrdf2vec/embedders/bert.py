from typing import List

import attr
from sklearn.utils.validation import check_is_fitted
from torch.utils.data import Dataset

from pyrdf2vec.embedders import Embedder
from pyrdf2vec.typings import Embeddings, Entities, SWalk

from transformers import (  # isort:skip
    DataCollatorForLanguageModeling,
    BertConfig,
    BertForMaskedLM,
    BertTokenizer,
    Trainer,
    TrainingArguments,
)


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


@attr.s
class BERT(Embedder):
    """Defines BERT embedding technique."""

    training_args = attr.ib(
        default=TrainingArguments(
            output_dir="./bert",
            overwrite_output_dir=True,
            num_train_epochs=3,
            warmup_steps=500,
            weight_decay=0.2,
            logging_dir="./logs",
            dataloader_num_workers=2,
            prediction_loss_only=True,
        ),
        validator=attr.validators.instance_of(TrainingArguments),
    )

    vocab_filename: str = attr.ib(
        default="bert-kg",
        validator=attr.validators.instance_of(str),
    )

    _vocabulary_size: int = attr.ib(
        init=False,
        repr=False,
        default=0,
        validator=attr.validators.instance_of(int),
    )

    def _build_vocabulary(
        self, nodes: List[str], is_update: bool = False
    ) -> None:
        """Build the BERT vocabulary with entities.

        Args:
            nodes: The nodes to build the vocabulary
            is_update: True if the new walks should be added to old model's
                walks, False otherwise.

        """
        with open(self.vocab_filename, "w") as f:
            if not is_update:
                for token in ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]:
                    f.write(f"{token}\n")
                    self._vocabulary_size += 1
            for node in nodes:
                f.write(f"{node}\n")
                self._vocabulary_size += 1

    def fit(self, walks: List[List[SWalk]], is_update: bool = False):
        """Fits the BERT model based on provided walks.

        Args:
            walks: The walks to create the corpus to to fit the model.
            is_update: True if the new walks should be added to old model's
                walks, False otherwise.
                Defaults to False.

        Returns:
            The fitted Word2Vec model.

        """
        walks = [walk for entity_walks in walks for walk in entity_walks]
        nodes = list({node for walk in walks for node in walk})
        self._build_vocabulary(nodes, is_update)
        self.tokenizer = BertTokenizer(
            vocab_file=self.vocab_filename,
            do_lower_case=False,
            never_split=nodes,
        )
        self.model_ = BertForMaskedLM(
            BertConfig(
                vocab_size=self._vocabulary_size,
                max_position_embeddings=512,
                type_vocab_size=1,
            )
        )
        Trainer(
            model=self.model_,
            args=self.training_args,
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer
            ),
            train_dataset=WalkDataset(walks, self.tokenizer),
        ).train()
        return self

    def transform(self, entities: Entities) -> Embeddings:
        """The features vector of the provided entities.

            Args:
                entities: The entities including test entities to create the
                embeddings. Since RDF2Vec is unsupervised, there is no label
                leakage.

        Returns:
            The features vector of the provided entities.

        """
        check_is_fitted(self, ["model_"])
        return [
            self.model_.bert.embeddings.word_embeddings.weight[
                self.tokenizer(entity)["input_ids"][1]
            ]
            .cpu()
            .detach()
            .numpy()
            for entity in entities
        ]
