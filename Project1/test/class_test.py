import numpy as np

from experiment_script import TokenIndicator, LogisticRegressionClassifier, TextClassifier
from tokenizers import Tokenizer


def test_token_indicator():
    tokenizer = TokenIndicator(length=5, verbose=True)
    token_id = [0, 1, 1, 0, 2]

    representation = tokenizer(token_id)

    expected_representation = [1, 1, 1, 0, 0]
    assert np.array_equal(representation, expected_representation), f"Expected {expected_representation}, but got {representation}"

def test_trivial():
    assert True, "This test should always pass"


def test_tokenizer():
    tokenizer = Tokenizer.from_pretrained("bert-base-cased")
    text = "Hello, how are you?"
    # get the token ids for the text
    tokens = tokenizer.encode(text).ids
    print(f"One Example token is: {tokens[0]}")

    assert isinstance(tokens, list), "Tokens should be a list"
    assert len(tokens) > 0, "Tokens list should not be empty"

