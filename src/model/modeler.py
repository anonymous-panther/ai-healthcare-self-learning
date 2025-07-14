import os
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import logging
from typing import Tuple, List, Any, Optional
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
import warnings
import json
from numpy.lib.stride_tricks import sliding_window_view
from .evaluator import ModelEvaluator

# Suppress TensorFlow warnings about compiled metrics
warnings.filterwarnings("ignore", message=".*Compiled the loaded model.*")
warnings.filterwarnings("ignore", message=".*compiled metrics have yet to be built.*")


class ModelTrainer:
    """
    Class for training LSTM models for hypoglycemia prediction.

    This class handles the complete model training pipeline including:
    - Sequence generation from glucose data
    - LSTM model building and compilation
    - Model training with early stopping
    - Model saving and metadata generation

    Note: Model evaluation is now handled by the separate ModelEvaluator class.
    """

    # Model hyperparameters
    SEQUENCE_LENGTH = 12  # 12 readings * 5 min/reading = 60 minutes of history
    LSTM_UNITS = 50
    DROPOUT_RATE = 0.2
    EPOCHS = 5
    BATCH_SIZE = 64

    # Column names
    COLUMN_TIME = "time"
    COLUMN_GLUCOSE_SCALED = "gl_scaled"
    COLUMN_HYPOGLYCEMIA_FUTURE = "is_hypo_event_in_future"
    COLUMN_PATIENT_ID = "id"

    # Model configuration
    LSTM_ACTIVATION = "relu"
    DENSE_ACTIVATION = "relu"
    OUTPUT_ACTIVATION = "sigmoid"
    OPTIMIZER = "adam"
    LOSS_FUNCTION = "binary_crossentropy"
    METRICS = ["accuracy"]
    DENSE_HIDDEN_UNITS = 25

    # Training configuration
    VALIDATION_SPLIT = 0.1
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    EARLY_STOPPING_PATIENCE = 5
    EARLY_STOPPING_MONITOR = "val_loss"

    # Artifacts directory and file paths
    ARTIFACTS_DIR = "artifacts"
    TRAINING_HISTORY_PLOT_FILE = os.path.join(ARTIFACTS_DIR, "training_history.png")
    MODEL_SAVE_FILE = os.path.join(ARTIFACTS_DIR, "hypoglycemia_prediction_model.keras")
    MODEL_METADATA_FILE = os.path.join(ARTIFACTS_DIR, "model_metadata.json")

    # Plot configuration
    TRAINING_HISTORY_FIGSIZE = (12, 6)
    PLOT_DPI = 300
    PLOT_BBOX_INCHES = "tight"

    # Plot labels and titles
    MODEL_ACCURACY_TITLE = "Model Accuracy"
    MODEL_LOSS_TITLE = "Model Loss"
    ACCURACY_LABEL = "Accuracy"
    LOSS_LABEL = "Loss"
    EPOCH_LABEL = "Epoch"
    TRAIN_LEGEND = "Train"
    VALIDATION_LEGEND = "Validation"
    UPPER_LEFT_LOC = "upper left"

    # Error messages and print statements
    NOT_ENOUGH_DATA_ERROR = "Not enough data to create sequences. Check SEQUENCE_LENGTH and available patient data."
    TRAINING_MSG = "\nTraining the LSTM model for {} epochs with batch size {}..."
    MODEL_TRAINING_COMPLETE_MSG = "\nModel training complete."
    TRAINING_HISTORY_SAVED_MSG = "Training history plot saved as: {}"
    MODEL_SAVED_MSG = "Model saved as: {}"
    MODEL_METADATA_SAVED_MSG = "Model metadata saved as: {}"

    # Model metadata
    MODEL_ARCHITECTURE_DESCRIPTION = "LSTM with Dense layers"

    def __init__(self):
        """
        Initialize the ModelTrainer.

        Sets up instance variables for storing model and data at different stages
        of the training pipeline.
        """
        self.model: Optional[Sequential] = None
        self.X_train: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None
        self.history: Optional[Any] = None

        os.makedirs(self.ARTIFACTS_DIR, exist_ok=True)

    def _create_sequences(
        self, data: pd.DataFrame, sequence_length: int
    ) -> Tuple[Any, Any]:
        """
        Create sequences using numpy stride_tricks for maximum performance.

        This method creates sliding windows of glucose values to generate
        input sequences for the LSTM model. It uses numpy's stride_tricks
        for efficient memory usage and performance.

        Args:
            data (pd.DataFrame): DataFrame with 'gl_scaled' and 'is_hypo_event_in_future' columns
            sequence_length (int): Length of input sequences to create

        Returns:
            Tuple[Any, Any]: A tuple containing:
                - X: numpy array of shape (n_samples, sequence_length)
                - y: numpy array of shape (n_samples,)
        """
        # Sort data by time to ensure chronological order
        data = data.sort_values(by=self.COLUMN_TIME)

        # Extract glucose values as numpy array
        gl_values = data[self.COLUMN_GLUCOSE_SCALED].values

        # Create sliding windows using numpy stride_tricks for efficiency
        # This creates overlapping windows of size sequence_length
        X = sliding_window_view(np.array(gl_values), sequence_length)

        # Extract target values starting from sequence_length position
        # This aligns targets with the end of each sequence
        y = data[self.COLUMN_HYPOGLYCEMIA_FUTURE].values[sequence_length - 1 :]

        # Ensure X and y have matching lengths
        min_length = min(len(X), len(y))
        X = X[:min_length]
        y = y[:min_length]

        return X, y

    def _create_sequences_from_patients(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences from all patients in the dataframe.

        This method processes each patient's data separately to create
        sequences, ensuring that sequences don't cross patient boundaries.

        Args:
            df (pd.DataFrame): DataFrame containing patient data

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                - X_sequences: Concatenated sequences from all patients
                - y_labels: Corresponding target labels

        Raises:
            ValueError: If no patients have enough data for sequences
        """
        all_X, all_y = [], []

        for patient_id, patient_df in df.groupby(self.COLUMN_PATIENT_ID):
            # Only process patients with sufficient data for at least one sequence
            if len(patient_df) > self.SEQUENCE_LENGTH:
                patient_X, patient_y = self._create_sequences(
                    patient_df, self.SEQUENCE_LENGTH
                )
                all_X.append(patient_X)
                all_y.append(patient_y)

        if all_X and all_y:
            X_sequences = np.concatenate(all_X)
            y_labels = np.concatenate(all_y)
        else:
            raise ValueError(self.NOT_ENOUGH_DATA_ERROR)

        return X_sequences, y_labels

    def _reshape_for_lstm(self, X_sequences: np.ndarray) -> np.ndarray:
        """
        Reshape sequences for LSTM input format.

        LSTM expects input in 3D format: (samples, timesteps, features).
        Since we only have one feature (glucose), we reshape to add
        the feature dimension.

        Args:
            X_sequences (np.ndarray): Input sequences of shape (samples, timesteps)

        Returns:
            np.ndarray: Reshaped sequences of shape (samples, timesteps, 1)
        """
        return X_sequences.reshape(X_sequences.shape[0], X_sequences.shape[1], 1)

    def _split_data(
        self, X_sequences: np.ndarray, y_labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into training and testing sets with proper type safety.

        This method uses stratified sampling to maintain class distribution
        in both training and testing sets, which is important for imbalanced
        datasets like hypoglycemia prediction.

        Args:
            X_sequences (np.ndarray): Input sequences
            y_labels (np.ndarray): Target labels

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
                - X_train: Training sequences
                - X_test: Testing sequences
                - y_train: Training labels
                - y_test: Testing labels
        """
        # Split data with stratification to maintain class distribution
        X_train, X_test, y_train, y_test = train_test_split(
            X_sequences,
            y_labels,
            test_size=self.TEST_SIZE,
            random_state=self.RANDOM_STATE,
            stratify=y_labels,
        )

        # Ensure all returned values are numpy arrays for type safety
        # Using np.asarray to avoid unnecessary copying if already numpy arrays
        X_train = np.asarray(X_train)
        X_test = np.asarray(X_test)
        y_train = np.asarray(y_train)
        y_test = np.asarray(y_test)

        return X_train, X_test, y_train, y_test

    def _generate_sequences(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate sequences from dataframe and split into train/test sets.

        This method orchestrates the complete sequence generation and
        data splitting process, providing detailed logging of shapes
        at each step.

        Args:
            df (pd.DataFrame): Input DataFrame with processed glucose data

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
                - X_train: Training sequences
                - X_test: Testing sequences
                - y_train: Training labels
                - y_test: Testing labels
        """
        # Create sequences from all patients
        X_sequences, y_labels = self._create_sequences_from_patients(df)

        # Reshape for LSTM input format
        X_sequences = self._reshape_for_lstm(X_sequences)

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = self._split_data(X_sequences, y_labels)

        return X_train, X_test, y_train, y_test

    def _create_lstm_layers(self, X_train: np.ndarray) -> Sequential:
        """
        Create the LSTM model architecture.

        This method builds a sequential LSTM model with the following architecture:
        - LSTM layer with specified units and activation
        - Dropout layer for regularization
        - Dense hidden layer for additional feature learning
        - Dropout layer for regularization
        - Output layer with sigmoid activation for binary classification

        Args:
            X_train (np.ndarray): Training data to determine input shape

        Returns:
            Sequential: Compiled LSTM model
        """
        model = Sequential(
            [
                # Input layer with explicit shape
                Input(shape=(X_train.shape[1], X_train.shape[2])),
                # LSTM layer - return_sequences=False for final timestep output only
                LSTM(
                    self.LSTM_UNITS,
                    activation=self.LSTM_ACTIVATION,
                ),
                # Dropout for regularization to prevent overfitting
                Dropout(self.DROPOUT_RATE),
                # Dense hidden layer for additional feature learning
                Dense(self.DENSE_HIDDEN_UNITS, activation=self.DENSE_ACTIVATION),
                # Dropout for regularization
                Dropout(self.DROPOUT_RATE),
                # Output layer: 1 neuron for binary classification
                # Sigmoid activation outputs probability between 0 and 1
                Dense(1, activation=self.OUTPUT_ACTIVATION),
            ]
        )
        return model

    def _compile_model(self, model: Sequential) -> None:
        """
        Compile the model with appropriate optimizer, loss, and metrics.

        This method configures the model for training with:
        - Adam optimizer for efficient gradient descent
        - Binary crossentropy loss for binary classification
        - Accuracy metric for performance monitoring

        Args:
            model (Sequential): The LSTM model to compile
        """
        model.compile(
            optimizer=self.OPTIMIZER, loss=self.LOSS_FUNCTION, metrics=self.METRICS
        )

    def _build_model(self, X_train: np.ndarray) -> Sequential:
        """
        Build and compile the LSTM model for hypoglycemia prediction.

        This method creates the model architecture, compiles it, and
        displays a summary of the model structure and parameters.

        Args:
            X_train (np.ndarray): Training data to determine input shape

        Returns:
            Sequential: Compiled LSTM model ready for training
        """
        # Create model architecture
        model = self._create_lstm_layers(X_train)

        # Compile the model
        self._compile_model(model)

        # Display model summary
        model.summary()
        return model

    def _create_early_stopping(self) -> EarlyStopping:
        """
        Create early stopping callback to prevent overfitting.

        This method configures early stopping to monitor validation loss
        and restore the best weights when training stops improving.

        Returns:
            EarlyStopping: Configured early stopping callback
        """
        return EarlyStopping(
            monitor=self.EARLY_STOPPING_MONITOR,
            patience=self.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
        )

    def _train_model_with_history(
        self, model: Sequential, X_train: np.ndarray, y_train: np.ndarray
    ) -> Any:
        """
        Train the model with early stopping and validation.

        This method trains the model with validation split and early stopping
        to prevent overfitting and ensure optimal performance.

        Args:
            model (Sequential): The LSTM model to train
            X_train (np.ndarray): Training sequences
            y_train (np.ndarray): Training labels

        Returns:
            Any: Training history containing loss and metrics
        """
        early_stopping = self._create_early_stopping()

        logging.info(self.TRAINING_MSG.format(self.EPOCHS, self.BATCH_SIZE))
        history = model.fit(
            X_train,
            y_train,
            epochs=self.EPOCHS,
            batch_size=self.BATCH_SIZE,
            # Use portion of training data for validation during training
            validation_split=self.VALIDATION_SPLIT,
            # Apply early stopping to prevent overfitting
            callbacks=[early_stopping],
            # Show progress during training
            verbose=1,
        )

        logging.info(self.MODEL_TRAINING_COMPLETE_MSG)
        return history

    def _plot_training_history(self, history: Any) -> None:
        """
        Plot and save training history.

        This method creates a comprehensive visualization of the training
        process, showing both accuracy and loss curves for training and
        validation sets.

        Args:
            history (Any): Training history from model.fit()
        """
        # Create figure for training history plots
        plt.figure(figsize=self.TRAINING_HISTORY_FIGSIZE)

        # Plot training & validation accuracy values
        plt.subplot(1, 2, 1)
        plt.plot(history.history["accuracy"])
        plt.plot(history.history["val_accuracy"])
        plt.title(self.MODEL_ACCURACY_TITLE)
        plt.ylabel(self.ACCURACY_LABEL)
        plt.xlabel(self.EPOCH_LABEL)
        plt.legend([self.TRAIN_LEGEND, self.VALIDATION_LEGEND], loc=self.UPPER_LEFT_LOC)

        # Plot training & validation loss values
        plt.subplot(1, 2, 2)
        plt.plot(history.history["loss"])
        plt.plot(history.history["val_loss"])
        plt.title(self.MODEL_LOSS_TITLE)
        plt.ylabel(self.LOSS_LABEL)
        plt.xlabel(self.EPOCH_LABEL)
        plt.legend([self.TRAIN_LEGEND, self.VALIDATION_LEGEND], loc=self.UPPER_LEFT_LOC)

        plt.tight_layout()
        plt.savefig(
            self.TRAINING_HISTORY_PLOT_FILE,
            dpi=self.PLOT_DPI,
            bbox_inches=self.PLOT_BBOX_INCHES,
        )
        plt.close()
        logging.info(
            self.TRAINING_HISTORY_SAVED_MSG.format(self.TRAINING_HISTORY_PLOT_FILE)
        )

    def _save_model_and_metadata(
        self, model: Sequential, X_train: np.ndarray, X_test: np.ndarray
    ) -> None:
        """
        Save the trained model and metadata.

        This method saves both the trained model and detailed metadata
        including hyperparameters and dataset information for future reference.

        Args:
            model (Sequential): The trained LSTM model
            X_train (np.ndarray): Training sequences
            X_test (np.ndarray): Testing sequences
        """
        # Save the trained model
        model.save(self.MODEL_SAVE_FILE)
        logging.info(self.MODEL_SAVED_MSG.format(self.MODEL_SAVE_FILE))

        # Save model metadata for future reference
        model_info = {
            "sequence_length": self.SEQUENCE_LENGTH,
            "lstm_units": self.LSTM_UNITS,
            "dropout_rate": self.DROPOUT_RATE,
            "epochs": self.EPOCHS,
            "batch_size": self.BATCH_SIZE,
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "model_architecture": self.MODEL_ARCHITECTURE_DESCRIPTION,
        }

        with open(self.MODEL_METADATA_FILE, "w") as f:
            json.dump(model_info, f, indent=2)
        logging.info(self.MODEL_METADATA_SAVED_MSG.format(self.MODEL_METADATA_FILE))

    def _train_model(
        self,
        model: Sequential,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> None:
        """
        Train the model and save results.

        This method orchestrates the complete training process including
        model training, history plotting, and model/metadata saving.

        Args:
            model (Sequential): The LSTM model to train
            X_train (np.ndarray): Training sequences
            y_train (np.ndarray): Training labels
            X_test (np.ndarray): Testing sequences
            y_test (np.ndarray): Testing labels
        """
        self.history = self._train_model_with_history(model, X_train, y_train)
        self._plot_training_history(self.history)
        self._save_model_and_metadata(model, X_train, X_test)

    def train_save_and_evaluate_model(self, df: pd.DataFrame) -> None:
        """
        Execute the complete model training pipeline.

        This method orchestrates the entire model training process from
        data preprocessing to final evaluation, providing a simple interface
        for the complete pipeline.

        Performs the following steps:
        1. Generate sequences from DataFrame
        2. Build and compile LSTM model
        3. Train the model with early stopping
        4. Save the model and metadata
        5. Evaluate the model using ModelEvaluator

        Args:
            df (pd.DataFrame): Input DataFrame with processed glucose data

        Returns:
            Sequential: Trained LSTM model ready for predictions
        """
        # Step 1: Generate sequences from processed data
        X_train, X_test, y_train, y_test = self._generate_sequences(df)

        # Step 2: Build and compile the LSTM model
        self.model = self._build_model(X_train)

        # Step 3: Train the model with evaluation
        self._train_model(self.model, X_train, y_train, X_test, y_test)

        # Step 4: Evaluate model performance using ModelEvaluator
        evaluator = ModelEvaluator()
        evaluator.evaluate_model(self.model, X_test, y_test)
