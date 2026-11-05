import numpy as np
import pandas as pd

from tokenizers import Tokenizer

from experiment_script import Encoder, ModelComparison, ProbabilisticModel, TextClassifier, LogisticRegressionClassifier, TokenIndicator, BagOfWords, Bert_Encoder, NeuralNetworkClassifier
from dataset import my_load_dataset

def small_test():
    """
    A small test to check if the classes are working as expected.
    """
    # Create a small dataset
    data = {
        'text': ['I love this movie', 'This movie is terrible', 'I hate this movie', 'This movie is great'],
        'class1': [1, 1, 0, 0],
        'class2': [0, 0, 1, 0],
        'class3': [0, 0, 0, 1]
    }
    # text class1 class2 class3
    # blab  0       1     0 
    df = pd.DataFrame(data)

    # Instantiate the classes
    tokenizer = Tokenizer.from_pretrained("bert-base-cased")

    # use token ids instead of tokens for the encoder
    vocab_size = tokenizer.get_vocab_size()
    encoder = TokenIndicator(length=vocab_size, verbose=True)
    model = LogisticRegressionClassifier()
    classifier = TextClassifier(tokenizer, encoder, model, df, verbose=True)

    # Run the experiment
    results = classifier.run_experiment(number_of_runs=5, number_of_folds=2)
    print("Experiment Results:", results)
    # plot the results
    classifier.visualize_results()

def test_experiment_comparison():
    a = np.random.randint(0, 10, size=(15))
    b = np.random.randint(5, 10, size=(15))
    c = np.random.randint(0, 5, size=(15))
    from experiment_script import ModelComparison

    example_dict = {"Random Model": a, "Logistic Regression": b, "Random Forest": c}
    comparison = ModelComparison(example_dict)
    comparison.visualize_results()

def test_logistic_regression_classifier():
    data, info = my_load_dataset()
    # choose only the first 100 rows for testing
    data = data.head(300)
    # use pretrained tokenizer
    tokenizer = Tokenizer.from_pretrained("bert-base-cased")
    identifier = TokenIndicator(length=tokenizer.get_vocab_size())
    bow_encoder = BagOfWords(length=tokenizer.get_vocab_size())
    bert_encoder = Bert_Encoder() 
    encoders = [identifier, bow_encoder, bert_encoder]
    model = LogisticRegressionClassifier()
    results_dict = {}
    encoder_names = ["TokenIndicator", "BagOfWords", "Bert_Encoder"]
    for i, encoder in enumerate(encoders):
        classifier = TextClassifier(tokenizer, encoder, model, data)
        results = classifier.run_experiment(number_of_runs=3, number_of_folds=2)
        print(f"Results for encoder {encoder.__class__.__name__}: {results}")
        results_dict[encoder_names[i]] = results

    # compare the results of the different encoders
    assert isinstance(results_dict, dict), "Results should be a dictionary"
    comparison = ModelComparison(experiment_results = results_dict)
    comparison.visualize_results()

def test_logistic_single_layer_MLP():
    data, info = my_load_dataset()
    # choose only the first 100 rows for testing
    data = data.head(300)
    # use pretrained tokenizer
    tokenizer = Tokenizer.from_pretrained("bert-base-cased")
    identifier = TokenIndicator(length=tokenizer.get_vocab_size())
    bow_encoder = BagOfWords(length=tokenizer.get_vocab_size())
    bert_encoder = Bert_Encoder() 
    encoders = [identifier, bow_encoder, bert_encoder]
    model = NeuralNetworkClassifier(activation="relu", solver="adam")
    results_dict = {}
    encoder_names = ["TokenIndicator", "BagOfWords", "Bert_Encoder"]
    for i, encoder in enumerate(encoders):
        classifier = TextClassifier(tokenizer, encoder, model, data)
        results = classifier.run_experiment(number_of_runs=3, number_of_folds=2)
        print(f"Results for encoder {encoder.__class__.__name__}: {results}")
        results_dict[encoder_names[i]] = results

    # compare the results of the different encoders
    assert isinstance(results_dict, dict), "Results should be a dictionary"
    comparison = ModelComparison(experiment_results = results_dict)
    comparison.visualize_results()




if __name__ == "__main__":
    test_logistic_single_layer_MLP()