import os
import json
import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import Tuple, Any, Optional
from tensorflow.keras.models import Sequential
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)
import seaborn as sns


class ModelEvaluator:
    """
    Class for evaluating LSTM models for hypoglycemia prediction.

    This class handles the complete model evaluation pipeline including:
    - Model prediction generation
    - Performance metric calculation
    - Visualization of results (confusion matrix, ROC curve)
    """

    # Evaluation configuration
    PREDICTION_THRESHOLD = 0.5

    # Artifacts directory and file paths
    ARTIFACTS_DIR = "artifacts"
    CONFUSION_MATRIX_PLOT_FILE = os.path.join(ARTIFACTS_DIR, "confusion_matrix.png")
    ROC_CURVE_PLOT_FILE = os.path.join(ARTIFACTS_DIR, "roc_curve.png")

    # Plot configuration
    CONFUSION_MATRIX_FIGSIZE = (6, 5)
    ROC_CURVE_FIGSIZE = (8, 6)
    PLOT_DPI = 300
    PLOT_BBOX_INCHES = "tight"

    # Plot labels and titles
    CONFUSION_MATRIX_TITLE = "Confusion Matrix"
    ACTUAL_LABEL = "Actual Label"
    PREDICTED_LABEL = "Predicted Label"
    PREDICTED_0_LABEL = "Predicted 0"
    PREDICTED_1_LABEL = "Predicted 1"
    ACTUAL_0_LABEL = "Actual 0"
    ACTUAL_1_LABEL = "Actual 1"
    ROC_CURVE_TITLE = "Receiver Operating Characteristic (ROC) Curve"
    FALSE_POSITIVE_RATE_LABEL = "False Positive Rate"
    TRUE_POSITIVE_RATE_LABEL = "True Positive Rate"
    LOWER_RIGHT_LOC = "lower right"

    # Error messages and print statements
    MODEL_EVALUATION_HEADER = "\n--- Model Evaluation ---"
    CONFUSION_MATRIX_SAVED_MSG = "Confusion matrix plot saved as: {}"
    ROC_CURVE_SAVED_MSG = "ROC curve plot saved as: {}"
    ROC_CURVE_LABEL = "ROC curve (area = {:.2f})"

    def __init__(self):
        """
        Initialize the ModelEvaluator.

        Sets up the artifacts directory for saving evaluation plots.
        """
        os.makedirs(self.ARTIFACTS_DIR, exist_ok=True)

    def _get_predictions(
        self, model: Sequential, X_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get model predictions and convert to binary predictions.

        This method generates both probability predictions and binary
        predictions using a threshold for classification.

        Args:
            model (Sequential): The trained LSTM model
            X_test (np.ndarray): Test sequences

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                - y_pred_proba: Probability predictions
                - y_pred_binary: Binary predictions (0 or 1)
        """
        y_pred_proba = model.predict(X_test).flatten()
        y_pred_binary = (y_pred_proba > self.PREDICTION_THRESHOLD).astype(int)
        return y_pred_binary, y_pred_proba

    def _calculate_metrics(
        self, y_test: np.ndarray, y_pred_binary: np.ndarray, y_pred_proba: np.ndarray
    ) -> Tuple[float, float, float, float, float]:
        """
        Calculate various evaluation metrics.

        This method computes comprehensive evaluation metrics including
        accuracy, precision, recall, F1-score, and ROC AUC.

        Args:
            y_test (np.ndarray): True labels
            y_pred_binary (np.ndarray): Binary predictions
            y_pred_proba (np.ndarray): Probability predictions

        Returns:
            Tuple[float, float, float, float, float]: A tuple containing:
                - accuracy: Overall accuracy
                - precision: Precision score
                - recall: Recall score
                - f1: F1-score
                - roc_auc: ROC AUC score
        """
        accuracy = float(accuracy_score(y_test, y_pred_binary))
        precision = float(precision_score(y_test, y_pred_binary))
        recall = float(recall_score(y_test, y_pred_binary))
        f1 = float(f1_score(y_test, y_pred_binary))
        roc_auc = float(roc_auc_score(y_test, y_pred_proba))
        return accuracy, precision, recall, f1, roc_auc

    def _save_metrics(
        self,
        accuracy: float,
        precision: float,
        recall: float,
        f1: float,
        roc_auc: float,
    ) -> None:
        """
        Save evaluation metrics to file and display them.

        This method saves all calculated metrics to a file in the artifacts directory
        and also displays them in the console.

        Args:
            accuracy (float): Overall accuracy
            precision (float): Precision score
            recall (float): Recall score
            f1 (float): F1-score
            roc_auc (float): ROC AUC score
        """
        # Create metrics dictionary with rounded values
        metrics = {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(roc_auc, 4),
        }

        # Save metrics to file
        metrics_file = os.path.join(self.ARTIFACTS_DIR, "model_metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)

        logging.info(f"Metrics saved to: {metrics_file}")

    def _plot_confusion_matrix(
        self, y_test: np.ndarray, y_pred_binary: np.ndarray
    ) -> None:
        """
        Plot and save confusion matrix.

        This method creates a heatmap visualization of the confusion matrix
        to show the performance of the binary classification model.

        Args:
            y_test (np.ndarray): True labels
            y_pred_binary (np.ndarray): Binary predictions
        """
        cm = confusion_matrix(y_test, y_pred_binary)
        plt.figure(figsize=self.CONFUSION_MATRIX_FIGSIZE)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=[self.PREDICTED_0_LABEL, self.PREDICTED_1_LABEL],
            yticklabels=[self.ACTUAL_0_LABEL, self.ACTUAL_1_LABEL],
        )
        plt.title(self.CONFUSION_MATRIX_TITLE)
        plt.ylabel(self.ACTUAL_LABEL)
        plt.xlabel(self.PREDICTED_LABEL)
        plt.savefig(
            self.CONFUSION_MATRIX_PLOT_FILE,
            dpi=self.PLOT_DPI,
            bbox_inches=self.PLOT_BBOX_INCHES,
        )
        plt.close()
        logging.info(
            self.CONFUSION_MATRIX_SAVED_MSG.format(self.CONFUSION_MATRIX_PLOT_FILE)
        )

    def _plot_roc_curve(
        self, y_test: np.ndarray, y_pred_proba: np.ndarray, roc_auc: float
    ) -> None:
        """
        Plot and save ROC curve.

        This method creates a ROC curve visualization to show the
        trade-off between true positive rate and false positive rate.

        Args:
            y_test (np.ndarray): True labels
            y_pred_proba (np.ndarray): Probability predictions
            roc_auc (float): ROC AUC score
        """
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        plt.figure(figsize=self.ROC_CURVE_FIGSIZE)
        plt.plot(fpr, tpr, label=self.ROC_CURVE_LABEL.format(roc_auc))
        # Add diagonal line representing random classifier
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel(self.FALSE_POSITIVE_RATE_LABEL)
        plt.ylabel(self.TRUE_POSITIVE_RATE_LABEL)
        plt.title(self.ROC_CURVE_TITLE)
        plt.legend(loc=self.LOWER_RIGHT_LOC)
        plt.grid(True)
        plt.savefig(
            self.ROC_CURVE_PLOT_FILE,
            dpi=self.PLOT_DPI,
            bbox_inches=self.PLOT_BBOX_INCHES,
        )
        plt.close()
        logging.info(self.ROC_CURVE_SAVED_MSG.format(self.ROC_CURVE_PLOT_FILE))

    def evaluate_model(
        self, model: Sequential, X_test: np.ndarray, y_test: np.ndarray
    ) -> Tuple[float, float, float, float, float]:
        """
        Evaluate the model and generate performance plots.

        This method performs comprehensive model evaluation including
        prediction generation, metric calculation, and visualization.

        Args:
            model (Sequential): The trained LSTM model
            X_test (np.ndarray): Test sequences
            y_test (np.ndarray): Test labels

        Returns:
            Tuple[float, float, float, float, float]: A tuple containing:
                - accuracy: Overall accuracy
                - precision: Precision score
                - recall: Recall score
                - f1: F1-score
                - roc_auc: ROC AUC score
        """
        logging.info(self.MODEL_EVALUATION_HEADER)

        # Generate predictions
        y_pred_binary, y_pred_proba = self._get_predictions(model, X_test)

        # Calculate evaluation metrics
        accuracy, precision, recall, f1, roc_auc = self._calculate_metrics(
            y_test, y_pred_binary, y_pred_proba
        )

        # Display metrics
        self._save_metrics(accuracy, precision, recall, f1, roc_auc)

        # Generate evaluation plots
        self._plot_confusion_matrix(y_test, y_pred_binary)
        self._plot_roc_curve(y_test, y_pred_proba, roc_auc)

        return accuracy, precision, recall, f1, roc_auc
