import numpy as np
import os
import pickle
import pandas as pd
import logging
from typing import Optional, Any
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model


class Predictor:
    """
    Class for making hypoglycemia predictions using trained models.

    This class handles the complete prediction pipeline including:
    - Loading saved models and scalers
    - Preprocessing glucose sequences
    - Making predictions with proper shape handling
    - Validation of model and scaler files
    """

    ARTIFACTS_DIR = "artifacts"
    # Default file paths
    DEFAULT_MODEL_PATH = os.path.join(
        ARTIFACTS_DIR, "hypoglycemia_prediction_model.keras"
    )
    DEFAULT_SCALER_PATH = os.path.join(ARTIFACTS_DIR, "glucose_scaler.pkl")

    # Column names
    GLUCOSE_COLUMN = "gl"

    # Messages
    MSG_MODEL_LOADED = "Model loaded successfully from: {}"
    MSG_SCALER_LOADED = "Glucose scaler loaded successfully from: {}"
    MSG_MODEL_NOT_FOUND = "Model file not found: {}"
    MSG_SCALER_NOT_FOUND = "Scaler file not found: {}"
    MSG_MODEL_FOUND = "✅ Model found: {}"
    MSG_MODEL_NOT_FOUND_UI = "❌ Model not found: {}"
    MSG_MODEL_TRAINING_REQUIRED = (
        "Please run the training script first to generate the model."
    )
    MSG_SCALER_FOUND = "✅ Scaler found: {}"
    MSG_SCALER_NOT_FOUND_UI = "❌ Scaler not found: {}"
    MSG_SCALER_TRAINING_REQUIRED = (
        "Please run the training script first to generate the scaler."
    )

    def __init__(
        self, model_path: Optional[str] = None, scaler_path: Optional[str] = None
    ):
        """
        Initialize the Predictor.

        Sets up the predictor with model and scaler paths, and initializes
        instance variables for storing loaded models and scalers.

        Args:
            model_path (str, optional): Path to the saved model file
            scaler_path (str, optional): Path to the saved scaler file
        """
        self.model_path = model_path or self.DEFAULT_MODEL_PATH
        self.scaler_path = scaler_path or self.DEFAULT_SCALER_PATH
        self.model: Optional[Sequential] = None
        self.scaler: Optional[Any] = None

    def _load_saved_model(self, model_path: Optional[str] = None) -> Sequential:
        """
        Load a saved model for future predictions.

        This method loads a trained Keras model from the specified path
        and validates that the file exists before loading.

        Args:
            model_path (str, optional): Path to the saved model file.
                If None, uses the instance model_path.

        Returns:
            Sequential: Loaded Keras model

        Raises:
            FileNotFoundError: If the model file does not exist
        """
        path = model_path or self.model_path

        if not os.path.exists(path):
            raise FileNotFoundError(self.MSG_MODEL_NOT_FOUND.format(path))

        model = load_model(path)
        logging.info(self.MSG_MODEL_LOADED.format(path))
        return model

    def _load_scaler(self, scaler_path: Optional[str] = None):
        """
        Load the saved glucose scaler.

        This method loads a saved scaler (typically MinMaxScaler) from
        the specified path and validates that the file exists.

        Args:
            scaler_path (str, optional): Path to the saved scaler file.
                If None, uses the instance scaler_path.

        Returns:
            object: Loaded scaler (typically MinMaxScaler)

        Raises:
            FileNotFoundError: If the scaler file does not exist
        """
        path = scaler_path or self.scaler_path

        if not os.path.exists(path):
            raise FileNotFoundError(self.MSG_SCALER_NOT_FOUND.format(path))

        with open(path, "rb") as f:
            scaler = pickle.load(f)
        logging.info(self.MSG_SCALER_LOADED.format(path))
        return scaler

    def _preprocess_glucose_sequence(self, glucose_sequence: np.ndarray) -> np.ndarray:
        """
        Preprocess glucose sequence for prediction.

        This method scales the glucose sequence using the loaded scaler
        and ensures the correct shape for LSTM model prediction.

        Args:
            glucose_sequence (np.ndarray): Raw glucose sequence

        Returns:
            np.ndarray: Preprocessed glucose sequence ready for prediction
        """
        # Scale the glucose sequence using the same scaler used during training
        # Convert to DataFrame with correct column name to avoid feature name warning
        glucose_df = pd.DataFrame(glucose_sequence.reshape(-1, 1))
        glucose_df.columns = [self.GLUCOSE_COLUMN]
        glucose_sequence_scaled = self.scaler.transform(glucose_df).flatten()

        # Ensure correct shape for prediction (1, sequence_length, 1)
        if glucose_sequence_scaled.ndim == 1:
            glucose_sequence_scaled = glucose_sequence_scaled.reshape(1, -1, 1)
        elif glucose_sequence_scaled.ndim == 2:
            glucose_sequence_scaled = glucose_sequence_scaled.reshape(
                1, glucose_sequence_scaled.shape[0], 1
            )

        return glucose_sequence_scaled

    def _make_prediction(self, glucose_sequence_scaled: np.ndarray) -> float:
        """
        Make prediction using the loaded model.

        This method performs the actual prediction using the preprocessed
        glucose sequence and returns the probability of hypoglycemia.

        Args:
            glucose_sequence_scaled (np.ndarray): Preprocessed glucose sequence

        Returns:
            float: Prediction probability (0-1)
        """
        prediction = self.model.predict(glucose_sequence_scaled, verbose=0)
        return prediction[0][0]

    def _check_model_exists(self, model_path: Optional[str] = None) -> bool:
        """
        Check if the saved model file exists.

        This method validates the existence of the model file and provides
        appropriate user feedback.

        Args:
            model_path (str, optional): Path to the model file.
                If None, uses the instance model_path.

        Returns:
            bool: True if model exists, False otherwise
        """
        path = model_path or self.model_path

        if os.path.exists(path):
            logging.info(self.MSG_MODEL_FOUND.format(path))
            return True
        else:
            logging.warning(self.MSG_MODEL_NOT_FOUND_UI.format(path))
            logging.warning(self.MSG_MODEL_TRAINING_REQUIRED)
            return False

    def _check_scaler_exists(self, scaler_path: Optional[str] = None) -> bool:
        """
        Check if the saved scaler file exists.

        This method validates the existence of the scaler file and provides
        appropriate user feedback.

        Args:
            scaler_path (str, optional): Path to the scaler file.
                If None, uses the instance scaler_path.

        Returns:
            bool: True if scaler exists, False otherwise
        """
        path = scaler_path or self.scaler_path

        if os.path.exists(path):
            logging.info(self.MSG_SCALER_FOUND.format(path))
            return True
        else:
            logging.warning(self.MSG_SCALER_NOT_FOUND_UI.format(path))
            logging.warning(self.MSG_SCALER_TRAINING_REQUIRED)
            return False

    def _validate_files_exist(self) -> bool:
        """
        Validate that both model and scaler files exist.

        This method checks for the existence of both required files
        and provides comprehensive feedback to the user.

        Returns:
            bool: True if both files exist, False otherwise
        """
        model_exists = self._check_model_exists()
        scaler_exists = self._check_scaler_exists()
        return model_exists and scaler_exists

    def _load_model_and_scaler(self) -> None:
        """
        Load both model and scaler for prediction.

        This method loads the trained model and scaler, caching them
        in instance variables for efficient repeated predictions.
        """
        self.model = self._load_saved_model()
        self.scaler = self._load_scaler()

    def execute(self, glucose_sequence: np.ndarray) -> float:
        """
        Execute the complete prediction pipeline.

        This method orchestrates the entire prediction process:
        1. Validates that required files exist
        2. Loads model and scaler (if not already loaded)
        3. Preprocesses the glucose sequence
        4. Makes the prediction

        Args:
            glucose_sequence (np.ndarray): Array of glucose values for prediction

        Returns:
            float: Prediction probability (0-1) indicating hypoglycemia risk

        Raises:
            FileNotFoundError: If model or scaler files are missing
            ValueError: If glucose sequence is invalid
        """
        # Validate input
        if glucose_sequence is None or len(glucose_sequence) == 0:
            raise ValueError("Glucose sequence cannot be empty")

        # Check if files exist
        if not self._validate_files_exist():
            raise FileNotFoundError("Required model or scaler files are missing")

        # Load model and scaler if not already loaded
        if self.model is None or self.scaler is None:
            self._load_model_and_scaler()

        # Preprocess the glucose sequence
        glucose_sequence_scaled = self._preprocess_glucose_sequence(glucose_sequence)

        # Make prediction
        prediction = self._make_prediction(glucose_sequence_scaled)

        return prediction
