import numpy as np
import pandas as pd

from tokenizers import Tokenizer

from experiment_script import Encoder, ProbabilisticModel, TextClassifier, LogisticRegressionClassifier, TokenIndicator


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

if __name__ == "__main__":
    test_experiment_comparison()