from abc import ABC, abstractmethod
from tokenizers import Tokenizer

from sklearn.model_selection import train_test_split

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
    def __call__(self, x: str) -> 'np.array':
        """Represent x as a D-dimensional vector of real numbers"""
        raise NotImplementedError("Concrete implementations will be discussed later")

class ProbabilisticModel(ABC):
    """
    A probability model is able to map input vector to a probability mass function. 
    """
    @abstractmethod
    def __call__(self, x: 'np.array') -> 'np.array':
        raise NotImplementedError("Concrete implementations will be discussed later")

class TextClassifier():

    def __init__(self, tokenizer:Tokenizer, encoder:Encoder, model:ProbabilisticModel, dataset):
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
        self.dataset = self.prepare_data(dataset)

    def prepare_data(self, dataset, train_size=0.8, val_size=0.1, test_size=0.1):
        """
        - Splits the data into train, validation and test sets.
        - 
        Args:
            dataset (_type_): _description_
            train_size (float): The proportion of the dataset to include in the train split.
            val_size (float): The proportion of the dataset to include in the validation split.
            test_size (float): The proportion of the dataset to include in the test split.
        """

        train_set, temp_set = train_test_split(dataset, train_size=train_size, random_state=42)

        val_set, test_set = train_test_split(temp_set, train_size=val_size/(val_size + test_size), random_state=42)

        tokenized_train_set = self.tokenizer.encode_batch(train_set['text'].tolist())

        return train_set, val_set, test_set

    def predict(self, text:str) -> 'np.array':
        pass

    def train(self, dataset):
        pass

    def execute_experiment(self, dataset):
        # Define fix metric here, e.g. accuracy, F1 score, etc.
        pass

    def visualize_results(self):
        pass
