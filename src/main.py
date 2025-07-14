#!/usr/bin/env python3
"""
Unified Hypoglycemia Prediction Pipeline

This script orchestrates the entire pipeline:
1. Dataset download and extraction
2. Data preprocessing and model training
3. Interactive prediction with user input

Usage:
    python src/main.py
"""

import os
import sys
import numpy as np
import pickle
import warnings
import logging

# Suppress SSL warnings
warnings.filterwarnings("ignore", message=".*OpenSSL.*")
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")

# Configure SSL for urllib3
try:
    import urllib3

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except ImportError:
    pass

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging to show INFO level messages in terminal
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# Import our modules
from dataset_retrieval.downloader import DatasetDownloader
from dataset_retrieval.csv_transformer import CsvTransformer
from model.dataframe_loader import DataframeLoader
from model.modeler import ModelTrainer
from predict.predictor import Predictor
from sklearn.preprocessing import MinMaxScaler

# Constants for file paths
DATASET_DIR = "dataset"
CSV_DATA_DIR = "csv_data"
CSV_FILE = "csv_data/lynch2022.csv"
ARTIFACTS_DIR = "artifacts"
MODEL_FILE = os.path.join(ARTIFACTS_DIR, "hypoglycemia_prediction_model.keras")
SCALER_FILE = os.path.join(ARTIFACTS_DIR, "glucose_scaler.pkl")

# Constants for download messages
MSG_DATASET_NOT_FOUND = "📥 Dataset not found. Starting download process..."
MSG_DOWNLOAD_CANCELLED = "❌ Download cancelled by user."
MSG_DOWNLOAD_FAILED = "❌ Failed to download dataset."
MSG_DOWNLOAD_SUCCESS = "✅ Dataset successfully downloaded and extracted to: {}"

# Constants for transform messages
MSG_TRANSFORMING_DATASET = "🔄 Transforming dataset from SAS to CSV format..."
MSG_TRANSFORM_SUCCESS = "✅ Dataset transformed successfully. Shape: {}"
MSG_TRANSFORM_ERROR = "❌ Error transforming dataset: {}"

# Constants for training messages
MSG_TRAINING_MODEL = "🤖 Training hypoglycemia prediction model..."
MSG_TRAINING_LSTM = "🔄 Training LSTM model..."
MSG_TRAINING_SUCCESS = "✅ Model training completed successfully!"
MSG_TRAINING_ERROR = "❌ Error training model: {}"
MSG_SCALER_SAVED = "✅ Glucose scaler saved as: {}"

# Constants for prediction messages
MSG_PREDICTION_MODE = "\n🎯 Starting prediction mode..."
MSG_GLUCOSE_INPUT_PROMPT = "\n📊 Enter glucose values (12 readings, 0-500 mg/dL):"
MSG_GLUCOSE_INPUT_EXAMPLE = "Enter values separated by spaces or commas (e.g., 120 118 115 112 110 108 105 102 100 98 95 92)"
MSG_GLUCOSE_SEQUENCE_PROMPT = "Glucose sequence: "
MSG_INVALID_VALUES_COUNT = "❌ Please enter exactly 12 glucose values."
MSG_INVALID_GLUCOSE_RANGE = "❌ Glucose value {} is out of range (0-500 mg/dL)."
MSG_INVALID_INPUT = "❌ Invalid input. Please enter numeric values only."
MSG_INPUT_CANCELLED = "\n❌ Input cancelled."
MSG_PREDICTION_ERROR = "❌ Error making prediction: {}"
MSG_PREDICTION_FAILED = "❌ Failed to make prediction."
MSG_PREDICTION_RESULTS = "\n📊 Prediction Results:"
MSG_GLUCOSE_SEQUENCE = "  Glucose sequence: {}"
MSG_HYPOGLYCEMIA_PROBABILITY = "  Hypoglycemia probability: {:.3f} ({:.1f}%)"
MSG_RISK_LEVEL = "  Risk level: {}"
MSG_HIGH_RISK_WARNING = "  ⚠️  WARNING: High hypoglycemia risk detected!"
MSG_MODERATE_RISK_CAUTION = "  ⚡ CAUTION: Moderate hypoglycemia risk"
MSG_NORMAL_GLUCOSE = "  ✅ Normal glucose levels"
MSG_CONTINUE_PREDICTION_PROMPT = "\nMake another prediction? (y/n): "
MSG_GOODBYE = "\n👋 Goodbye!"

# Constants for main execution messages
MSG_SYSTEM_TITLE = "🏥 Hypoglycemia Prediction System"
MSG_DATASET_NOT_FOUND_MAIN = "📁 Dataset not found."
MSG_DOWNLOAD_PROMPT = "Download dataset? (y/n): "
MSG_DATASET_REQUIRED = "❌ Dataset required. Exiting."
MSG_DATASET_FOUND = "✅ Dataset found."
MSG_CSV_NOT_FOUND = "📄 Processed CSV not found."
MSG_TRANSFORM_PROMPT = "Transform dataset to CSV? (y/n): "
MSG_PROCESSED_DATA_REQUIRED = "❌ Processed data required. Exiting."
MSG_CSV_FOUND = "✅ Processed CSV found."
MSG_NO_MODEL_FOUND = "🤖 No trained model found."
MSG_TRAIN_MODEL_PROMPT = "Train model? (y/n): "
MSG_MODEL_REQUIRED = "❌ Trained model required. Exiting."
MSG_MODEL_FOUND = "✅ Trained model found."
MSG_RETRAIN_MODEL_PROMPT = "Retrain model? (y/n): "
MSG_UNEXPECTED_ERROR = "❌ Unexpected error: {}"

# Constants for thresholds and values
GLUCOSE_READINGS_COUNT = 12
GLUCOSE_MIN_VALUE = 0
GLUCOSE_MAX_VALUE = 500
PREDICTION_ERROR_VALUE = -1
RISK_THRESHOLDS = {
    "LOW": 0.3,
    "MEDIUM": 0.7,
    "HIGH_WARNING": 0.5,
    "MODERATE_WARNING": 0.3,
}

# Constants for risk levels
RISK_LEVELS = {"ERROR": "ERROR", "LOW": "LOW", "MEDIUM": "MEDIUM", "HIGH": "HIGH"}

# Constants for user input validation
USER_INPUT_YES = ["y", "yes"]
SEPARATOR_LINE = "=" * 60

# Constants for data processing
GLUCOSE_COLUMN = "gl"
SCALED_GLUCOSE_COLUMN = "gl_scaled"


class HypoglycemiaPredictor:
    """Main class to orchestrate the entire hypoglycemia prediction pipeline."""

    def __init__(self):
        self.dataset_dir = DATASET_DIR
        self.csv_data_dir = CSV_DATA_DIR
        self.csv_file = CSV_FILE
        self.model_file = MODEL_FILE
        self.scaler_file = SCALER_FILE

    def _check_dataset_exists(self) -> bool:
        """Check if the dataset has been downloaded and extracted."""
        return os.path.exists(self.dataset_dir) and os.path.isdir(self.dataset_dir)

    def _check_csv_exists(self) -> bool:
        """Check if the processed CSV file exists."""
        return os.path.exists(self.csv_file)

    def _check_model_exists(self) -> bool:
        """Check if a trained model exists."""
        return os.path.exists(self.model_file)

    def _download_dataset(self) -> bool:
        """Download and extract the dataset."""
        logging.info(MSG_DATASET_NOT_FOUND)
        logging.info(SEPARATOR_LINE)

        # Download the dataset using the new class-based approach
        downloader = DatasetDownloader()
        success, zip_path, extract_path = downloader.get_dataset()

        if success:
            logging.info(MSG_DOWNLOAD_SUCCESS.format(extract_path))
            return True
        else:
            logging.error(MSG_DOWNLOAD_FAILED)
            return False

    def _transform_dataset(self) -> bool:
        """Transform the SAS dataset to CSV format."""
        logging.info(MSG_TRANSFORMING_DATASET)
        logging.info(SEPARATOR_LINE)

        try:
            transformer = CsvTransformer()
            df = transformer.generate_dataframe()
            logging.info(MSG_TRANSFORM_SUCCESS.format(df.shape))
            return True
        except Exception as e:
            logging.error(MSG_TRANSFORM_ERROR.format(e))
            return False

    def _train_model(self) -> bool:
        """Train the hypoglycemia prediction model."""
        logging.info(MSG_TRAINING_MODEL)
        logging.info(SEPARATOR_LINE)

        try:
            # Load and preprocess data using the new class-based approach
            loader = DataframeLoader()
            df_with_target = loader.load_dataframe(self.csv_file)

            # Scale glucose values
            scaler = MinMaxScaler()
            df_with_target[SCALED_GLUCOSE_COLUMN] = scaler.fit_transform(
                df_with_target[[GLUCOSE_COLUMN]]
            )

            # Create artifacts directory if it doesn't exist
            os.makedirs(ARTIFACTS_DIR, exist_ok=True)

            # Save the scaler
            with open(self.scaler_file, "wb") as f:
                pickle.dump(scaler, f)
            logging.info(MSG_SCALER_SAVED.format(self.scaler_file))

            # Train model using the new class-based approach
            logging.info(MSG_TRAINING_LSTM)
            trainer = ModelTrainer()
            trainer.train_save_and_evaluate_model(df_with_target)

            logging.info(MSG_TRAINING_SUCCESS)
            return True

        except Exception as e:
            logging.error(MSG_TRAINING_ERROR.format(e))
            return False

    def _get_glucose_sequence(self) -> np.ndarray:
        """Get glucose sequence input from user."""
        logging.info(MSG_GLUCOSE_INPUT_PROMPT)
        logging.info(MSG_GLUCOSE_INPUT_EXAMPLE)

        while True:
            try:
                user_input = input(MSG_GLUCOSE_SEQUENCE_PROMPT).strip()

                # Parse input (handle both spaces and commas)
                values = user_input.replace(",", " ").split()

                if len(values) != GLUCOSE_READINGS_COUNT:
                    logging.warning(MSG_INVALID_VALUES_COUNT)
                    continue

                # Convert to float and validate range
                glucose_values = []
                for val in values:
                    glucose = float(val)
                    if glucose < GLUCOSE_MIN_VALUE or glucose > GLUCOSE_MAX_VALUE:
                        logging.warning(MSG_INVALID_GLUCOSE_RANGE.format(glucose))
                        continue
                    glucose_values.append(glucose)

                if len(glucose_values) == GLUCOSE_READINGS_COUNT:
                    return np.array(glucose_values)

            except ValueError:
                logging.warning(MSG_INVALID_INPUT)
            except KeyboardInterrupt:
                logging.info(MSG_INPUT_CANCELLED)
                sys.exit(1)

    def _make_prediction(self, glucose_sequence: np.ndarray) -> float:
        """Make prediction using the trained model."""
        try:
            predictor = Predictor(self.model_file, self.scaler_file)
            prediction = predictor.execute(glucose_sequence)
            return prediction

        except Exception as e:
            logging.error(MSG_PREDICTION_ERROR.format(e))
            return PREDICTION_ERROR_VALUE

    def _interpret_prediction(self, prediction: float) -> str:
        """Interpret the prediction probability."""
        if prediction < 0:
            return RISK_LEVELS["ERROR"]
        elif prediction < RISK_THRESHOLDS["LOW"]:
            return RISK_LEVELS["LOW"]
        elif prediction < RISK_THRESHOLDS["MEDIUM"]:
            return RISK_LEVELS["MEDIUM"]
        else:
            return RISK_LEVELS["HIGH"]

    def _run_prediction_loop(self):
        """Run the interactive prediction loop."""
        logging.info(MSG_PREDICTION_MODE)
        logging.info(SEPARATOR_LINE)

        while True:
            try:
                # Get glucose sequence
                glucose_sequence = self._get_glucose_sequence()

                # Make prediction
                prediction = self._make_prediction(glucose_sequence)

                if prediction >= 0:
                    risk_level = self._interpret_prediction(prediction)

                    logging.info(MSG_PREDICTION_RESULTS)
                    logging.info(MSG_GLUCOSE_SEQUENCE.format(glucose_sequence))
                    logging.info(
                        MSG_HYPOGLYCEMIA_PROBABILITY.format(
                            prediction, prediction * 100
                        )
                    )
                    logging.info(MSG_RISK_LEVEL.format(risk_level))

                    if prediction > RISK_THRESHOLDS["HIGH_WARNING"]:
                        logging.warning(MSG_HIGH_RISK_WARNING)
                    elif prediction > RISK_THRESHOLDS["MODERATE_WARNING"]:
                        logging.warning(MSG_MODERATE_RISK_CAUTION)
                    else:
                        logging.info(MSG_NORMAL_GLUCOSE)
                else:
                    logging.error(MSG_PREDICTION_FAILED)

                # Ask if user wants to continue
                continue_prediction = (
                    input(MSG_CONTINUE_PREDICTION_PROMPT).strip().lower()
                )
                if continue_prediction not in USER_INPUT_YES:
                    break

            except KeyboardInterrupt:
                logging.info(MSG_GOODBYE)
                break

    def run(self):
        """Main execution method."""
        logging.info(MSG_SYSTEM_TITLE)
        logging.info(SEPARATOR_LINE)

        # Step 1: Check and download dataset if needed
        if not self._check_dataset_exists():
            logging.info(MSG_DATASET_NOT_FOUND_MAIN)
            if input(MSG_DOWNLOAD_PROMPT).strip().lower() in USER_INPUT_YES:
                if not self._download_dataset():
                    logging.error(MSG_DOWNLOAD_FAILED)
                    return
            else:
                logging.error(MSG_DATASET_REQUIRED)
                return
        else:
            logging.info(MSG_DATASET_FOUND)

        # Step 2: Check and transform dataset if needed
        if not self._check_csv_exists():
            logging.info(MSG_CSV_NOT_FOUND)
            if input(MSG_TRANSFORM_PROMPT).strip().lower() in USER_INPUT_YES:
                if not self._transform_dataset():
                    logging.error(MSG_TRANSFORM_ERROR.format(""))
                    return
            else:
                logging.error(MSG_PROCESSED_DATA_REQUIRED)
                return
        else:
            logging.info(MSG_CSV_FOUND)

        # Step 3: Check and train model if needed
        if not self._check_model_exists():
            logging.info(MSG_NO_MODEL_FOUND)
            if input(MSG_TRAIN_MODEL_PROMPT).strip().lower() in USER_INPUT_YES:
                if not self._train_model():
                    logging.error(MSG_TRAINING_ERROR.format(""))
                    return
            else:
                logging.error(MSG_MODEL_REQUIRED)
                return
        else:
            logging.info(MSG_MODEL_FOUND)
            if input(MSG_RETRAIN_MODEL_PROMPT).strip().lower() in USER_INPUT_YES:
                if not self._train_model():
                    logging.error(MSG_TRAINING_ERROR.format(""))
                    return

        # Step 4: Start prediction mode
        self._run_prediction_loop()


def main():
    """Main function."""
    try:
        predictor = HypoglycemiaPredictor()
        predictor.run()
    except KeyboardInterrupt:
        logging.info(MSG_GOODBYE)
    except Exception as e:
        logging.error(MSG_UNEXPECTED_ERROR.format(e))


if __name__ == "__main__":
    main()
