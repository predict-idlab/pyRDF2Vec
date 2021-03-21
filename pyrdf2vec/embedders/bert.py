from typing import List

import attr
from pyrdf2vec.embedders import Embedder
from sklearn.utils.validation import check_is_fitted
from torch.utils.data import Dataset

from transformers import (  # isort:skip
    DataCollatorForLanguageModeling,
    DistilBertConfig,
    DistilBertForMaskedLM,
    DistilBertTokenizer,
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
        self.tokenizer = DistilBertTokenizer(
            vocab_file=self.vocab_filename,
            do_lower_case=False,
            never_split=unique_hops,
        )
        self.model_ = DistilBertForMaskedLM(
            DistilBertConfig(
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
            train_dataset=WalkDataset(corpus, self.tokenizer),
        ).train()
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
