import numpy as np
import torch
import transformers
from keras.callbacks import ReduceLROnPlateau
from keras.layers import BatchNormalization, Dense, Dropout
from keras.models import Sequential
from keras.optimizers import RMSprop

LR = 2e-5
EPOCHS = 3
BATCH_SIZE = 16
SEQ_LEN = 128


def attention_mask(padded: np.ndarray) -> np.ndarray:
    """Gets the attention mask.

    Returns:
        np.ndarray: The attention mask with the zeros to ignore used for padding.

    """
    return np.where(padded != 0, 1, 0)


def create_nn() -> Sequential:
    """Defines the architecture of the network.

    Returns:
        The untrained model.

    """
    model = Sequential()
    model.add(Dense(1000, input_shape=(773,), activation="relu"))
    model.add(BatchNormalization(axis=-1))
    model.add(Dense(256, activation="relu"))
    model.add(BatchNormalization(axis=-1))
    model.add(Dropout(0.25))
    model.add(Dense(256, activation="relu"))
    model.add(BatchNormalization(axis=-1))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation="relu"))
    model.add(BatchNormalization(axis=-1))
    model.add(Dropout(0.25))
    model.add(Dense(10, activation="relu"))
    model.add(BatchNormalization(axis=-1))
    model.add(Dropout(0.25))
    model.add(Dense(1, activation="sigmoid"))
    return model


def compile_nn(train_x, train_y) -> Sequential:
    """Compile the neural network

    Args:
      model: The model architecture.
      X_train (DataFrame): X Training set (scaled bert embeddings + attributes)
      y_train (Series):Y Training set (ONLY Spam indication)

    Returns:
        The trained model.

    """
    model.compile(
        optimizer=RMSprop(lr=LR),
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )

    fitted_network = model.fit(
        train_x,
        train_y,
        validation_split=0.2,
        batch_size=BATCH_SIZE,
        shuffle=True,
        epochs=EPOCHS,
        callbacks=[
            ReduceLROnPlateau(
                monitor="val_accuracy",
                patience=3,
                verbose=1,
                factor=0.5,
                min_lr=0.00001,
            )
        ],
    )
    return model


# "http://dl-learner.org/carcinogenesis#d133",
# = root node
#
# This is what walking strategies (+ sampling strategies) give us before
# Word2Vec is called. For now, I just take a list of 3 tuples, in a reality
# we have a list of MANY tuples of strings.
walks = [
    (
        "http://dl-learner.org/carcinogenesis#d133",
        "<pyrdf2vec.graphs.kg.Vertex object at 0x7f26268e0a30>",
        "b'}\\xba\\x99\\xf06k\\x8e\\x8b'",
        "<pyrdf2vec.graphs.kg.Vertex object at 0x7f2626132910>",
        "b'\\xd8\\xa3\\xca\\xc0\\x04\\xa9g\\xf1'",
    ),
    (
        "http://dl-learner.org/carcinogenesis#d312",
        "<pyrdf2vec.graphs.kg.Vertex object at 0x7f2620cb0130>",
        "b'\\xcf>\\x92t\\xed;*X'",
        "<pyrdf2vec.graphs.kg.Vertex object at 0x7f26231ca250>",
        "b'\\xd4\\xe0\\xfa\\xd8y\\xc9\\xf5\\xa5'",
    ),
    (
        "http://dl-learner.org/carcinogenesis#d133",
        "<pyrdf2vec.graphs.kg.Vertex object at 0x7f26247a0eb0>",
        "b'`\\nA\\x98{\\xc8\\xf1>'",
        "<pyrdf2vec.graphs.kg.Vertex object at 0x7f26281341c0>",
        "b'=\\x0b\\xf2_\\x95d\\xb1\\x0e'",
    ),
]

tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")

# Here, we going to tokenize and prepare for the model one or several
# sequence(s) [hops for our case] or one or several pair(s) of sequences.
#
# MORE: https://huggingface.co/transformers/main_classes/tokenizer.html?highlight=pretrainedtokenizer#transformers.PreTrainedTokenizerFast.__call__
#
# The "encode" function take only string, a list/tuple of strings (pretokenized
# string) or a list/tuple of integers. If a list of strings is given, we need to
# add "is_split_into_words=True" to the encode function.
#
# Here, we have a list of tuples of strings, I guess that we should go all over the
# tuples of strings and encode them individually.
#
# I am not sure here if we should use: padding=True and truncation=True here,
# as we encode each hop separately.
#
# If not padding, the output batch with sequences can have different lengths.
#
# If not truncation, the output batch with sequence lengths can be greater than the
# model maximum admissible input size.
#
# For any BERT model, the maximum sequence length after tokenization is 512, but
# we can go below. Here, I use padding="max_length" to pad to the maximum length
# of 512
#
# Each walk will have a length of 512 now.
#
walks_ids = [
    tokenizer.encode(walk, padding="max_length", is_split_into_words=True)
    for walk in walks
]
# print(walks_ids)
#
# This is what walks_ids looks like (a list of three sub-lists of ID.), I guess
# this is right.
#
# [[101, 8299, 1024, 1013, 1013, 21469, 1011, 4553, 2121, 1012, 8917, 1013, 2482,
# 21081, 23737, 1001, 1040, 17134, 2509, 1026, 1052, 12541, 20952, 2475, 3726,
# 2278, 1012, 19287, 1012, 4705, 1012, 19449, 4874, 2012, 1014, 2595, 2581, 2546,
# 23833, 23833, 2620, 2063, 2692, 2050, 14142, 1028, 1038, 1005, 1065, 1032, 1060,
# 3676, 1032, 1060, 2683, 2683, 1032, 1060, 2546, 2692, 2575, 2243, 1032, 1060,
# 2620, 2063, 1032, 1060, 2620, 2497, 1005, 1026, 1052, 12541, 20952, 2475, 3726,
# 2278, 1012, 19287, 1012, 4705, 1012, 19449, 4874, 2012, 1014, 2595, 2581, 2546,
# 23833, 23833, 17134, 24594, 10790, 1028, 1038, 1005, 1032, 1060, 2094, 2620,
# 1032, 1060, 2050, 2509, 1032, 1060, 3540, 1032, 1060, 2278, 2692, 1032, 1060,
# 2692, 2549, 1032, 1060, 2050, 2683, 2290, 1032, 1060, 2546, 2487, 1005, 102,
# 0, 0, ... ],
#
# [101, 8299, 1024, 1013, 1013, 21469, 1011, 4553, 2121, 1012, 8917, 1013, 2482,
# 21081, 23737, 1001, 1040, 21486, 2475, 1026, 1052, 12541, 20952, 2475, 3726,
# 2278, 1012, 19287, 1012, 4705, 1012, 19449, 4874, 2012, 1014, 2595, 2581, 2546,
# 23833, 11387, 27421, 24096, 14142, 1028, 1038, 1005, 1032, 1060, 2278, 2546,
# 1028, 1032, 1060, 2683, 2475, 2102, 1032, 1060, 2098, 1025, 1008, 1060, 1005,
# 1026, 1052, 12541, 20952, 2475, 3726, 2278, 1012, 19287, 1012, 4705, 1012,
# 19449, 4874, 2012, 1014, 2595, 2581, 2546, 23833, 21926, 2487, 3540, 17788,
# 2692, 1028, 1038, 1005, 1032, 1060, 2094, 2549, 1032, 1060, 2063, 2692, 1032,
# 1060, 7011, 1032, 1060, 2094, 2620, 2100, 1032, 1060, 2278, 2683, 1032, 1060,
# 2546, 2629, 1032, 1060, 2050, 2629, 1005, 102, 0, 0, ...],
#
# [101, 8299, 1024, 1013, 1013,
# 21469, 1011, 4553, 2121, 1012, 8917, 1013, 2482, 21081, 23737, 1001, 1040,
# 17134, 2509, 1026, 1052, 12541, 20952, 2475, 3726, 2278, 1012, 19287, 1012,
# 4705, 1012, 19449, 4874, 2012, 1014, 2595, 2581, 2546, 23833, 18827, 2581, 2050,
# 2692, 15878, 2692, 1028, 1038, 1005, 1036, 1032, 6583, 1032, 1060, 2683, 2620,
# 1063, 1032, 1060, 2278, 2620, 1032, 1060, 2546, 2487, 1028, 1005, 1026, 1052,
# 12541, 20952, 2475, 3726, 2278, 1012, 19287, 1012, 4705, 1012, 19449, 4874,
# 2012, 1014, 2595, 2581, 2546, 23833, 22407, 17134, 23632, 2278, 2692, 1028,
# 1038, 1005, 1027, 1032, 1060, 2692, 2497, 1032, 1060, 2546, 2475, 1035, 1032,
# 1060, 2683, 2629, 2094, 1032, 1060, 2497, 2487, 1032, 1060, 2692, 2063, 1005,
# 102, 0, 0, ...]]

model = transformers.BertModel.from_pretrained("bert-base-uncased")
with torch.no_grad():
    hidden_states = model(
        torch.tensor(walks_ids),
        torch.tensor(attention_mask(np.asarray(walks_ids))),
    )

print(hidden_states)
# We have this:
#
# (tensor([[[-0.1189, -0.1017, -0.1706,  ..., -0.3311,  0.1749,  0.6851],
#          [-0.3664, -0.6074, -0.4456,  ...,  0.4977,  0.5386,  0.4002],
#          [ 0.7369,  0.2549, -0.4688,  ...,  0.1034, -0.5025, -0.4412],
#          ...,
#          [ 0.0552,  0.1179,  0.4614,  ..., -0.0406,  0.0021, -0.3909],
#          [ 0.1850,  0.1115,  0.4711,  ..., -0.0453, -0.1338, -0.4545],
#          [ 0.0199, -0.0015,  0.4633,  ..., -0.2117, -0.1140, -0.3593]],

#         [[ 0.0184, -0.1338, -0.0925,  ..., -0.1408,  0.1836,  0.8528],
#          [-0.2635, -0.5687, -0.4438,  ...,  0.4877,  0.4482,  0.5123],
#          [ 0.7737,  0.2528, -0.4347,  ...,  0.0734, -0.5287, -0.4130],
#          ...,
#          [ 0.2165,  0.1605,  0.3566,  ..., -0.1463, -0.5508,  0.3348],
#          [ 0.2225, -0.1280,  0.3300,  ..., -0.0287, -0.1518, -0.1936],
#          [ 0.0433,  0.0119,  0.3646,  ..., -0.1605, -0.0873, -0.1983]],

#         [[ 0.0900, -0.0913, -0.0535,  ..., -0.1975,  0.1712,  0.8963],
#          [-0.1928, -0.4942, -0.4105,  ...,  0.4968,  0.5242,  0.4056],
#          [ 0.7697,  0.3269, -0.4774,  ...,  0.0133, -0.6275, -0.4058],
#          ...,
#          [ 0.2874,  0.2817,  0.4091,  ...,  0.0237,  0.2543,  0.0492],
#          [ 0.3075,  0.1741,  0.5251,  ..., -0.0631,  0.0084,  0.1626],
#          [ 0.0766, -0.0126,  0.3674,  ..., -0.2128, -0.0399, -0.2170]]]), tensor([[-0.5768, -0.5141, -0.9834,  ..., -0.9678, -0.6708,  0.4902],
#         [-0.2273, -0.2802, -0.9677,  ..., -0.9431, -0.5255,  0.2575],
#         [-0.3786, -0.4354, -0.9765,  ..., -0.9414, -0.6007,  0.3571]]))


# After that we have the hidden states, I suppose that we should create the
# architecture of our network and give the X_train and y_train to it. For
# pyRDF2Vec, I think we get that with:
#
# train_entities, train_labels = load_data(
#    DATASET["train"][0], DATASET["train"][1], DATASET["train"][2]
# )
#
# SEE: https://github.com/IBCNServices/pyRDF2Vec/blob/4894fc84d9cafd1fb8081889d4d8c19165afc328/examples/mutag.py#L89
#
# However, with Word2Vec, we didn't need to give the x_train and y_train.
#
# trained_model = compile_nn(create_nn(), x_train, y_train)
