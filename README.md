# Hypoglycemia Prediction System

A unified machine learning system for predicting hypoglycemic events in Type 1 Diabetes patients using LSTM neural networks.

## 🏥 Overview

This system provides a complete pipeline for:

1. **Dataset Management**: Downloading and processing the IOBP2 clinical trial dataset
2. **Model Training**: Training LSTM models on glucose time series data
3. **Interactive Prediction**: Making real-time predictions on user-provided glucose sequences

## 📁 Project Structure

```
assignment/
├── src/
│   ├── main.py                                   # Unified pipeline executable
│   ├── dataset_retrieval/
│   │   ├── downloader.py                         # Dataset download functionality
│   │   └── csv_transformer.py                    # SAS to CSV conversion
│   ├── model/
│   │   ├── modeler.py                            # LSTM model implementation
│   │   ├── dataframe_loader.py                   # Data preprocessing
│   │   └── evaluator.py                          # Model evaluation and metrics
│   ├── predict/
│   │   └── predictor.py                          # Prediction functionality
│   └── requirements.txt                          # Python dependencies
├── artifacts/                                    # Generated model files and visualizations
│   ├── hypoglycemia_prediction_model.keras       # Trained LSTM model
│   ├── glucose_scaler.pkl                        # Glucose value scaler
│   ├── model_metadata.json                       # Model configuration
│   ├── model_metrics.json                        # Model metrics
│   ├── confusion_matrix.png                      # Model evaluation visualization
│   ├── roc_curve.png                             # ROC curve visualization
│   └── training_history.png                      # Training history visualization
├── csv_data/                                     # Processed CSV files
│   └── lynch2022.csv                             # Transformed dataset
├── dataset/                                      # Downloaded dataset
│   ├── Data Tables in SAS/                       # Original SAS data files
│   ├── Data Tables/                              # Processed data tables
│   ├── CRFs/                                     # Case report forms
│   ├── IOBP_PIVOTAL_PROTOCOL_V8.5_26JAN2021.pdf  # Study protocol
│   ├── DataGlossary.rtf                          # Data dictionary
│   └── IOBP2 Dataset ReadMe.rtf                  # Dataset documentation
├── downloaded_dataset.zip                        # Original dataset archive
├── .gitignore                                    # Git ignore patterns
└── README.md                                     # This file
```

## 🚀 Quick Start

### Prerequisites

Install the required dependencies:

```bash
pip install -r src/requirements.txt
```

### System Requirements

- **Python**: 3.8 or higher
- **Memory**: At least 8GB RAM for model training
- **Storage**: 2GB free space for dataset and models
- **Internet**: Required for initial dataset download

### Running the System

Execute the unified system:

```bash
python src/main.py
```

The system will guide you through each step:

1. **Dataset Download**: If not present, prompts to download the IOBP2 dataset
2. **Data Processing**: Converts SAS files to CSV format
3. **Model Training**: Trains the LSTM model on glucose data
4. **Interactive Prediction**: Enter glucose sequences for real-time predictions

## 📊 Usage Examples

### Example 1: First-time Setup

```bash
$ python src/main.py

🏥 Hypoglycemia Prediction System
============================================================
📁 Dataset not found.
Download dataset? (y/n): y

📥 Dataset not found. Starting download process...
============================================================
✅ Dataset successfully downloaded and extracted to: dataset

📄 Processed CSV not found.
Transform dataset to CSV? (y/n): y
🔄 Transforming dataset from SAS to CSV format...
✅ Dataset transformed successfully. Shape: (1234567, 6)

🤖 No trained model found.
Train model? (y/n): y
🔄 Training LSTM model...
✅ Model training completed successfully!

🎯 Starting prediction mode...
============================================================

📊 Enter glucose values (12 readings, 0-500 mg/dL):
Enter values separated by spaces or commas (e.g., 120 118 115 112 110 108 105 102 100 98 95 92)
Glucose sequence: 120 118 115 112 110 108 105 102 100 98 95 92

📊 Prediction Results:
  Glucose sequence: [120 118 115 112 110 108 105 102 100  98  95  92]
  Hypoglycemia probability: 0.234 (23.4%)
  Risk level: LOW
  ✅ Normal glucose levels

Make another prediction? (y/n): n
```

### Example 2: Using Existing Model

```bash
$ python src/main.py

🏥 Hypoglycemia Prediction System
============================================================
✅ Dataset found.
✅ Processed CSV found.
✅ Trained model found.
Retrain model? (y/n): n

🎯 Starting prediction mode...
============================================================

📊 Enter glucose values (12 readings, 0-500 mg/dL):
Glucose sequence: 140 130 120 110 100 90 80 70 60 50 40 30

📊 Prediction Results:
  Glucose sequence: [140 130 120 110 100  90  80  70  60  50  40  30]
  Hypoglycemia probability: 0.856 (85.6%)
  Risk level: HIGH
  ⚠️  WARNING: High hypoglycemia risk detected!
```

## 🔧 System Components

### 1. Dataset Management (`src/dataset_retrieval/`)

- **`downloader.py`**: Downloads the IOBP2 clinical trial dataset from a public S3 link provided by JAEB
- **`csv_transformer.py`**: Converts SAS files to CSV format with patient demographics

### 2. Model Training (`src/model/`)

- **`modeler.py`**: LSTM model implementation for sequence prediction
- **`dataframe_loader.py`**: Data preprocessing and hypoglycemia event identification
- **`evaluator.py`**: Model evaluation, metrics calculation, and visualization

### 3. Prediction (`src/predict/`)

- **`predictor.py`**: Loads trained models and makes predictions on new data
  - Loads trained Keras models and glucose scalers
  - Makes predictions on glucose sequences
  - Handles model loading and prediction execution

## 📈 Model Architecture

The system uses an LSTM (Long Short-Term Memory) neural network:

- **Input**: 12 consecutive glucose readings (60 minutes of data)
- **Architecture**: LSTM → Dropout → Dense → Dropout → Dense → Sigmoid
- **Output**: Probability of hypoglycemia (0-1)

### Model Parameters

```python
SEQUENCE_LENGTH = 12  # 12 readings = 60 minutes
LSTM_UNITS = 50
DROPOUT_RATE = 0.2
EPOCHS = 5
BATCH_SIZE = 64
```

## 📊 Data Requirements

### Input Format

The system expects glucose sequences with:

- **Length**: Exactly 12 values
- **Range**: 0-500 mg/dL
- **Format**: Space or comma-separated values

### Example Inputs

```bash
# Normal glucose levels
120 118 115 112 110 108 105 102 100 98 95 92

# Declining glucose (moderate risk)
120 115 110 105 100 95 90 85 80 75 70 65

# Rapid decline (high risk)
140 130 120 110 100 90 80 70 60 50 40 30
```

## 🎯 Prediction Interpretation

### Risk Levels

- **LOW** (0-30%): Normal glucose levels
- **MEDIUM** (30-70%): Moderate hypoglycemia risk
- **HIGH** (70-100%): High hypoglycemia risk

### Warning Thresholds

- **⚠️ WARNING**: Probability > 50%
- **⚡ CAUTION**: Probability > 30%
- **✅ Normal**: Probability ≤ 30%

## 📁 Generated Files

The system creates several files during execution in the `artifacts/` directory:

- `hypoglycemia_prediction_model.keras`: Trained LSTM model (modern Keras format)
- `glucose_scaler.pkl`: Glucose value scaler
- `model_metadata.json`: Model configuration and hyperparameters
- `confusion_matrix.png`: Model evaluation visualization
- `roc_curve.png`: ROC curve visualization
- `training_history.png`: Training history visualization

### Modular Architecture

The system is designed with modular components that work together:

- **`src/main.py`**: Orchestrates the entire pipeline with refactored, concise code
- **`src/model/`**: Model training and data preprocessing modules
- **`src/predict/predictor.py`**: Prediction functionality
- **`src/dataset_retrieval/`**: Dataset management utilities

### Component Integration

All components are designed to be used through the main pipeline (`src/main.py`). The modular design allows for:

- **Easy maintenance**: Each component has a single responsibility
- **Code reusability**: Functions can be imported and used by other modules
- **Clean architecture**: Clear separation of concerns
- **Unified interface**: Single entry point for all functionality
- **Organized outputs**: All generated files are stored in the `artifacts/` directory

## 🐛 Troubleshooting

### Common Issues

1. **Dataset Download Fails**

   - Check internet connection
   - Ensure sufficient disk space

2. **Model Training Errors**

   - Check if TensorFlow is installed
   - Verify data format
   - Ensure sufficient memory

3. **Prediction Errors**
   - Verify model file exists
   - Check glucose value range (0-500)
   - Ensure exactly 12 values

### Error Messages

- `"Dataset required"`: Download the dataset first
- `"Trained model required"`: Train the model first
- `"Invalid input"`: Check glucose value format
- `"Out of range"`: Values must be 0-500 mg/dL

## 📚 Technical Details

### Dataset Information

- **Source**: IOBP2 Clinical Trial (JAEB)
- **Patients**: 440+ Type 1 Diabetes patients
- **Data**: Continuous glucose monitoring
- **Duration**: 13-week study period

### Data Preprocessing

1. **Resampling**: 5-minute intervals
2. **Scaling**: Min-max normalization (0-1)
3. **Sequence Creation**: Sliding windows of 12 readings
4. **Target Creation**: Future hypoglycemia events

## Acknowledgments

- **JAEB**: For providing the IOBP2 dataset
- **IOBP2 Study Team**: For conducting the clinical trial
- **Research Community**: For advancing diabetes technology

---

**Note**: This system is designed for research purposes. Clinical decisions should be made by healthcare professionals.
