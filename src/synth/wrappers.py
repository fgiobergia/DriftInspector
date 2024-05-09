
from skmultiflow.data import STAGGERGenerator, \
                             AGRAWALGenerator, \
                             LEDGenerator, \
                             SEAGenerator, \
                             HyperplaneGenerator
from sklearn.preprocessing import KBinsDiscretizer
import pandas as pd
import numpy as np


class Discretizable:
    def __init__(self, n_bins, **kwargs):
        self.kb = KBinsDiscretizer(encode="onehot-dense", n_bins=n_bins, strategy="uniform")
        self.kb_fit = False
        self.input_features = None
    
    def get_metadata(self, X):
        if not self.kb_fit:
            X_bin = self.kb.fit_transform(X).astype(bool)
            self.kb_fit = True
        else:
            X_bin = self.kb.transform(X).astype(bool)
        return pd.DataFrame(data=X_bin, columns=self.kb.get_feature_names_out(input_features=self.input_features))

class SEAWrapper(Discretizable):
    def __init__(self, random_state=42, noise_percentage=0.0):
        super().__init__(n_bins=5)

        function_from, function_to = np.random.choice(4, size=2, replace=False)
        
        self.stream=SEAGenerator(
            balance_classes=False,
            classification_function=function_from,
            noise_percentage=noise_percentage,
            random_state=random_state
        )

        self.drift_stream=SEAGenerator(
            balance_classes=False,
            classification_function=function_to,
            noise_percentage=noise_percentage,
            random_state=random_state
        )

class AgrawalWrapper(Discretizable):

    def __init__(self, random_state=42, perturbation=0.0):
        super().__init__(n_bins=5)

        function_from, function_to = 0, 2
        # function_from, function_to = np.random.choice(10, size=2, replace=False)

        self.stream=AGRAWALGenerator(
            balance_classes=False,
            classification_function=function_from,
            perturbation=perturbation,
            random_state=random_state
        )

        self.drift_stream=AGRAWALGenerator(
            balance_classes=False,
            classification_function=function_to,
            perturbation=perturbation,
            random_state=random_state
        )

        self.input_features = [
            "salary",
            "commission",
            "age",
            "elevel",
            "car",
            "zipcode",
            "hvalue",
            "hyears",
            "loan"
        ]

class HyperplaneWrapper(Discretizable):
    def __init__(self, random_state=42, noise_percentage=0.0):
        super().__init__(n_bins=5)

        self.stream = HyperplaneGenerator(
            random_state=random_state,
            noise_percentage=noise_percentage
        )
        
        self.drift_stream = HyperplaneGenerator(
            random_state=random_state+1,
            noise_percentage=noise_percentage
        )

# trivial -- will not be used
class LEDWrapper(Discretizable):
    def __init__(self, random_state=42, noise_percentage=0.0):
        super().__init__(n_bins=5)
        
        self.stream = LEDGenerator(
            random_state=random_state,
            noise_percentage=noise_percentage
        )
        
        self.drift_stream = LEDGenerator(
            random_state=random_state+1,
            noise_percentage=noise_percentage
        )

# trivial -- will not be used
class STAGGERWrapper(Discretizable):
    def __init__(self, random_state=42):
        super().__init__(n_bins=5)
        
        function_from, function_to = np.random.choice(3, size=2, replace=False)

        self.stream = STAGGERGenerator(
            random_state=random_state,
            classification_function=function_from
        )
        
        self.drift_stream = STAGGERGenerator(
            random_state=random_state,
            classification_function=function_to
        )