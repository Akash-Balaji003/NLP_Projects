import numpy as np
import pandas as pd
from experiment_script import TokenIndicator, LogisticRegressionClassifier, TextClassifier, Bert_Encoder
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

def test_bert_encoder():
    tokenizer = Tokenizer.from_pretrained("bert-base-cased")
    dataset = pd.DataFrame(
        {"text":["I am feeling bad", "I am fine, thank you!", "Today is the best day of my life", "I am not happy with the service", "This is a great product!"],
         "emotion":[0, 1, 1, 0, 1]}
    )

    encoder = Bert_Encoder()
    model = LogisticRegressionClassifier()
    text_classifier = TextClassifier(tokenizer=tokenizer, encoder=encoder, model=model, dataset=dataset)

    X = text_classifier.prep_dataset[0]
    assert isinstance(X, np.ndarray), "X should be a numpy array"
    # check the shape of X
    assert X.shape[1] == 768, f"Expected feature dimension to be 768, but got {X.shape[1]}"

    # conduct experiment
    # text_classifier.run_experiment()

