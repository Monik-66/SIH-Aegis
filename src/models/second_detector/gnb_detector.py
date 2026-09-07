import numpy as np
from sklearn.naive_bayes import GaussianNB


class GNBDetector:
    """
    Gaussian Naive Bayes classifier for MFCC-based
    real-vs-spoof detection.
    """

    def __init__(self):
        self.model = GaussianNB()
        self.is_fitted = False

    def fit(self, X, y):
        """
        Train the classifier.

        Parameters
        ----------
        X : np.ndarray
            Shape: (n_samples, n_features)

        y : np.ndarray
            Binary labels:
                0 = spoof
                1 = bonafide
        """

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        if X.ndim != 2:
            raise ValueError("X must be a 2D feature matrix.")

        if len(X) != len(y):
            raise ValueError(
                "Number of feature rows must match number of labels."
            )

        if len(np.unique(y)) < 2:
            raise ValueError(
                "Training requires at least two classes."
            )

        self.model.fit(X, y)
        self.is_fitted = True

    def predict_proba(self, features):
        """
        Return spoof/bonafide probabilities.
        """

        if not self.is_fitted:
            raise RuntimeError(
                "GNBDetector must be fitted before prediction."
            )

        features = np.asarray(
            features,
            dtype=np.float32
        ).reshape(1, -1)

        probabilities = self.model.predict_proba(
            features
        )[0]

        classes = self.model.classes_

        spoof_score = 0.0
        bonafide_score = 0.0

        for cls, probability in zip(
            classes,
            probabilities
        ):
            if cls == 0:
                spoof_score = float(probability)
            elif cls == 1:
                bonafide_score = float(probability)

        return {
            "spoof_score": spoof_score,
            "bonafide_score": bonafide_score,
        }

    def predict(self, features):
        """
        Return the final classification.
        """

        probabilities = self.predict_proba(features)

        prediction = (
            "SPOOF"
            if probabilities["spoof_score"]
            > probabilities["bonafide_score"]
            else "BONAFIDE"
        )

        return {
            "prediction": prediction,
            **probabilities,
        }
