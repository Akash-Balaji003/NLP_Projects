from abc import ABC, abstractmethod
from typing import Literal
import tqdm
from tokenizers import Tokenizer
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from tokenizers import Tokenizer
from transformers import AutoModel

from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from analysis import BenavoliCorrelatedT

class Encoder(ABC):
    """
    Class to convert a string into a vector representation. The vector representation is a D-dimensional vector of real numbers. The dimensionality of the representation space is given by the __len__ method.

    Args:
        ABC (_type_): _description_

    Raises:
        NotImplementedError: _description_
        NotImplementedError: _description_
    """

    @abstractmethod
    def __len__(self) -> int:
        """Dimensionality of the representation space"""
        raise NotImplementedError("Concrete implementations will be discussed later")

    @abstractmethod
    def __call__(self, x: list, **kwargs) -> 'np.ndarray':
        """Represent x as a D-dimensional vector of real numbers"""
        raise NotImplementedError("Concrete implementations will be discussed later")

class ProbabilisticModel(ABC):
    """
    A probability model is able to map input vector to a probability mass function. 
    """
    @abstractmethod
    def __call__(self, x: 'np.ndarray') -> 'np.ndarray':
        raise NotImplementedError("Concrete implementations will be discussed later")

    @abstractmethod
    def train(self, X_train: 'np.ndarray', y_train: 'np.ndarray'):
        raise NotImplementedError("Concrete implementations will be discussed later")

    @abstractmethod
    def cross_validate(self, X: 'np.ndarray', y: 'np.ndarray', k:int, metric:Literal["accuracy", "f1", "roc_auc"]="accuracy"):
        raise NotImplementedError("Concrete implementations will be discussed later")
        
class TextClassifier():

    def __init__(self, tokenizer:Tokenizer, encoder:Encoder, model:ProbabilisticModel, dataset:pd.DataFrame, verbose:bool=False):
        """

        Args:
            tokenizer (Tokenizer): tokenizer convert text into a sequence of tokens indicex
            encoder (Encoder): gets the sequence of tokens and converts it into a vector representation
            model (ProbabilisticModel): convert the vector representation into a probability mass function
            dataset (pandas.DataFrame (column names: text, one hot encoded labels - 28 columns): GoEmotion
        """
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.model = model
        self.dataset = dataset
        self.verbose = verbose
        self.prep_dataset = self.prepare_data(dataset)
        self.experiment_results = None
        self.metric_values = None

    def prepare_data(self, dataset)-> tuple['np.ndarray', 'np.ndarray']:
        """
        
        - tokenize
        - encode
        """

        vec_list = []
        X = dataset['text'].tolist()

        for text in tqdm.tqdm(X, desc="Preparing data"):
            if self.verbose:
                print(f"Processing text: {text}")
            encoding = self.tokenizer.encode(text)
            token_ids = encoding.ids
            if self.verbose:
                print(f"Token ids: {token_ids}")
            assert isinstance(token_ids, list), "tokenizer.encode should return a list of token ids"
            encoded_data = self.encoder(token_ids, encoding=encoding)
            if self.verbose:
                print(f"Encoded data: {encoded_data}")
            vec_list.append(encoded_data)

        # convert list of vectors into a numpy array
        encoded_data = np.array(vec_list)

        # convert one hot encoded labels into a single label column
        y = dataset["emotion"].astype(int)

        y = y.to_numpy()
        print(f"Encoded data shape: {encoded_data.shape}, Labels shape: {y.shape}")
        return encoded_data, y

    def predict(self, text:str) -> 'np.ndarray':
        """
        Predict the probability mass function for a given text.

        Args:
            text (str): The input text to classify.
        """
        token_ids = self.tokenizer.encode(text)
        assert isinstance(token_ids, list), "tokenizer.encode should return a list of token ids"
        encoded_data = self.encoder(token_ids)
        return self.model(encoded_data)

    def run_experiment(self, number_of_folds:int, number_of_runs:int, metric:Literal["accuracy", "f1", "roc_auc"]="accuracy"):
        """
        Execute a k-fold cross validation on the train and validation set,
        using "probably" a f1 - score metric. The k-fold cross validation is executed
        n times, such that the returned vector contains n*k measurements.
                

        Args:
            number_of_folds (int): The number of folds for cross-validation.
            number_of_runs (int): The number of times to run the cross-validation.
            metric (Literal["accuracy", "f1", "roc_auc"]): The metric to use for evaluation.
            dataset (_type_): _description_
        Returns:
            np.ndarray of shape (n*k, ) containing the measurements for each fold,  
        """
        self.metric_value = metric
        self.k = number_of_folds
        self.n = number_of_runs
        metric_values = []
        X = self.prep_dataset[0]
        y = self.prep_dataset[1]
        for n in tqdm.tqdm(range(number_of_runs), desc="Running experiments"):
            assert isinstance(X, np.ndarray), "Encoded data should be a numpy array"
            random_indices = np.random.permutation(len(X))
            shuffled_data = X[random_indices]
            shuffled_labels = y[random_indices]
            cv_scores = self.model.cross_validate(shuffled_data, shuffled_labels, k=number_of_folds, metric=metric)
            metric_values += list(cv_scores)
        self.experiment_results = np.array(metric_values)

        return np.array(metric_values)

    def get_experiment_results(self):
        if self.experiment_results is None:
            raise ValueError("Experiment results are not available. Please run execute_experiment first.")
        return self.experiment_results

    def visualize_results(self):
        """
        Visualize the results of the experiment.
        Using a two subplots
        - one shows a boxplot of the metric values
        - the other shows a histogram of the metric values
        Returns:
            None
        Raises:
            ValueError: If experiment results are not available.
        """
        if self.experiment_results is None:
            raise ValueError("Experiment results are not available. Please run execute_experiment first.")
        # use seaborn to visualize the results
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        sns.boxplot(data=self.experiment_results, ax=axes[0])
        sns.histplot(data=self.experiment_results, ax=axes[1])
        # set titles
        # set figure title
        fig.suptitle(f'Experiment Results: {self.n} runs of {self.k}-fold cross-validation', fontsize=16)
        axes[0].set_title(f'Boxplot of {self.metric_value} Values')
        axes[1].set_title(f'Histogram of {self.metric_value} Values')
        plt.show()

class TokenIndicator(Encoder):
    """
       TokenIndicator creates a vector out of one text x. The vector has the dimension of the 
       length of the vocabulary of the tokenizer. If one specific token appears in the text, 
       then the position of this token is set to 1. 
       -> information is only contain token (yes/no)
       -> no information about count of occurance 
       -> no information about ordering or relations

    Args:
        Encoder (_type_): _description_
    """

    def __init__(self, length: int, verbose: bool = False):
        self.length = length
        self.Verbose = verbose
        
    def __len__(self) -> int:
        """Dimensionality of the representation space"""
        return self.length

    def __call__(self, x: list, **kwargs) -> 'np.ndarray':
        """X is a list of list of token ids"""
        if self.Verbose:
            print(f"Encoding input: {x}")
        unique_token_ids = list(set(x))

        if self.Verbose:
            print(f"Unique token ids: {unique_token_ids}")

        # A 0 vector with the same length as the vocabulary except in positions corresponding 
        # to tokens appearing in the document
        vector = np.zeros(self.length)
        vector[unique_token_ids] = 1
        vector = vector.astype(int)  # Convert to integer type
        if self.Verbose:
            print(f"Final vector: {vector}")
        return vector


class BagOfWords(Encoder):
    """
       BagOfWords creates a vector out of one text x. The vector has the dimension of the 
       length of the vocabulary of the tokenizer. If one specific token appears in the text, 
       then the position of this token is set to the number of occurances of this token in the text. 
       -> information about count of occurance 
       -> no information about ordering or relations

    Args:
        Encoder (_type_): _description_
    """

    def __init__(self, length: int, verbose: bool = False):
        self.length = length
        self.Verbose = verbose
        
    def __len__(self) -> int:
        """Dimensionality of the representation space"""
        return self.length

    def __call__(self, x: list) -> 'np.ndarray':
        """X is a list of list of token ids"""
        if self.Verbose:
            print(f"Encoding input: {x}")
        # A 0 vector with the same length as the vocabulary except in positions corresponding 
        # to tokens appearing in the document
        vector = np.zeros(self.length)
        for token_id in x:
            vector[token_id] += 1
        vector = vector.astype(int)  # Convert to integer type
        if self.Verbose:
            print(f"Final vector: {vector}")
        return vector


class LogisticRegressionClassifier(ProbabilisticModel):
    """
    Use scikit-learn LogisticRegression to implement a logistic regression classifier.
    
    Args:
        ProbabilisticModel (_type_): _description_
    """
    def __init__(self, solver:Literal["newton-cg", "lbfgs", "liblinear", "sag", "saga"]="lbfgs", max_iter=100):
        self.solver = solver
        self.max_iter = max_iter
        self.model = LogisticRegression(solver=solver, max_iter=max_iter)

    def __call__(self, x: 'np.ndarray') -> 'np.ndarray':
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        return self.model.predict_proba(x)

    def train(self, X_train: 'np.ndarray', y_train: 'np.ndarray'):
        if self.model is None:
            raise ValueError("Model has not been initialized.")
        self.model.fit(X_train, y_train)

    def cross_validate(self, X: 'np.ndarray', y: 'np.ndarray', k:int, metric:Literal["accuracy", "f1", "roc_auc"]="accuracy"):
        if self.model is None:
            raise ValueError("Model has not been initialized.")
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(self.model, X, y, cv=k, scoring=metric)
        return scores

class NeuralNetworkClassifier(ProbabilisticModel):
    """
    Use scikit-learn MLPClassifier to implement a neural network classifier.
    
    Args:
        ProbabilisticModel (_type_): _description_
    """
    def __init__(self,
                  hidden_layer_sizes=(128,),
                  activation:Literal["identity", "logistic", "tanh", "relu"]="relu",
                  solver:Literal["adam", "lbfgs", "sgd"]="adam",
                  max_iter=200):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.max_iter = max_iter
        self.model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver, max_iter=max_iter)

    def __call__(self, x: 'np.ndarray') -> 'np.ndarray':
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        return self.model.predict_proba(x)

    def train(self, X_train: 'np.ndarray', y_train: 'np.ndarray'):
        if self.model is None:
            raise ValueError("Model has not been initialized.")
        self.model.fit(X_train, y_train)

    def cross_validate(self, X: 'np.ndarray', y: 'np.ndarray', k:int, metric:Literal["accuracy", "f1", "roc_auc"]="accuracy"):
        if self.model is None:
            raise ValueError("Model has not been initialized.")
        scores = cross_val_score(self.model, X, y, cv=k, scoring=metric)
        return scores

class ModelComparison:
    """
       A class to compare different model experiment results and
       visualize the posterior distribution.
    """

    def __init__(self, experiment_results:dict)->None:
        """
        For Model Comparison, we do not want to rely on the common
        student t-test, as this approach only considers the mean of the different groups,
        hence it ignores the variance and has no measurement of uncertainty. 
        Computing the posterior distribution of the experiment results for each model.
        The dataset given is list of tuples (a1, b1), (a2, b2), ..., (an, bn) 
        with the assumption that the data follows a multidimensional normal distribution.
        - Covariance Matrix: rho * diag(sigama)
        - prior distribution on the mean: student t
        - uniform prior on sigam

        P(mu, sigma | data) = P(data | mu, sigma) * P(mu, sigma) / P(data)
        Compute the MAP estimate of the posterior - or maybe full bayesian approach
        Args:
            experiment_results (dict): {model_name: np.ndarray of shape (n*k, ) containing the measurements for each fold}
        """
        self.experiment_results = experiment_results

    def visualize_results(self)->None:
        if len(self.experiment_results) == 2:
            posterior = BenavoliCorrelatedT(
                np.random.randint(low=70, high=80, size=100)/100,
                np.random.randint(low=40, high=60, size=100)/100,
                names=["logistic-indicator-enc", "mlp-indicator-enc"]
            )
            p_a, p_tie, p_b = posterior.probs(0.005)
            print(f"The ROPE (Region of Practical Equivalence) is set to 0.005, meaning that if the difference between the two models is less than 0.005, we consider them practically equivalent.")
            print(f"A=logistic_indicator-enc B=mlp_indicator-enc: P(A wins)={p_a:.4f} P(Tie)={p_tie:.4f} P(B wins)={p_b:.4f} Effect Size={posterior.effect_size:.2f}")
            _ = posterior.plot(0.005)

        if len(self.experiment_results) > 2:
            # In case of more than 2 models, we choose a simple workaround and
            # compute the posterior for each pair of models.
            all_posteriors = {}
            for i, model_a in enumerate(self.experiment_results.keys()):
                for j, model_b in enumerate(self.experiment_results.keys()):
                    if i < j:
                        posterior = BenavoliCorrelatedT(
                            self.experiment_results[model_a],
                            self.experiment_results[model_b],
                            names=[model_a, model_b]
                        )
                        p_a, p_tie, p_b = posterior.probs(0.005)
                        print(f"A={model_a} B={model_b}: P(A wins)={p_a:.4f} P(Tie)={p_tie:.4f} P(B wins)={p_b:.4f} Effect Size={posterior.effect_size:.2f}")
                        all_posteriors[(model_a, model_b)] = posterior

            # create a grid of subplots to visualize the posteriors 
            # for each pair of models n rows and one column
            num_model_comparisons = len(all_posteriors)
            for i, (model_a, model_b) in enumerate(all_posteriors.keys()):
                posterior = all_posteriors[(model_a, model_b)]
                posterior.plot(0.005)
            plt.show()
                
class Bert_Encoder(Encoder):

    def __init__(self):
        self.model = AutoModel.from_pretrained("bert-base-cased")
        self.model.eval()

    def __len__(self) -> int:
        """Dimensionality of the representation space"""
        return 768

    def __call__(self, x: list, **kwargs) -> 'np.ndarray':
        # convert the list of token ids into a tensor
        assert isinstance(x, list), "Input x should be a list of token ids"
        token_ids = torch.tensor(x).unsqueeze(0)  # shape (1, seq_len)
        encoding = kwargs.get('encoding', None)  # allow passing a precomputed encoding
        assert not encoding is None, "You must provide a precomputed encoding for the input text"


        attention_mask = torch.tensor([encoding.attention_mask])

        with torch.no_grad():
            output = self.model(input_ids=token_ids, attention_mask=attention_mask)

        token_vectors = output.last_hidden_state[0]
        mask = attention_mask.unsqueeze(-1)
        embedding = ((output.last_hidden_state * mask).sum(1) / mask.sum(1))[0]   # (768,)

        # convert embedding to numpy array
        embedding_np = embedding.numpy()
        return embedding_np