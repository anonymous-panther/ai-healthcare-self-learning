import pandas as pd
import logging
from typing import Optional


class DataframeLoader:
    """
    Class for loading and preprocessing dataframes for hypoglycemia prediction.

    This class handles the complete data preprocessing pipeline including:
    - Loading CSV data with time conversion
    - Resampling and imputation of glucose data
    - Creating target variables for hypoglycemia prediction
    """

    # Backend configuration
    MATPLOTLIB_BACKEND = "Agg"

    # Column names
    COLUMN_TIME = "time"
    COLUMN_ID = "id"
    COLUMN_GLUCOSE = "gl"
    COLUMN_IS_HYPO = "is_hypo"
    COLUMN_IS_HYPO_EVENT_IN_FUTURE = "is_hypo_event_in_future"

    # Data processing constants
    RESAMPLE_FREQ = "5min"
    HYPO_THRESHOLD = 70  # mg/dL
    PREDICTION_HORIZON_MINUTES = (
        60  # Predict if hypoglycemia occurs within the next X minutes
    )
    DEFAULT_NUM_ROWS = 200000

    # Interpolation and rolling window parameters
    INTERPOLATION_METHOD = "linear"
    ROLLING_CLOSED_PARAM = "right"
    ROLLING_MIN_PERIODS = 1

    # Data type conversions
    FILL_VALUE = 0
    DATA_TYPE_INT = "int"

    # Messages for data processing
    MSG_CREATING_TARGET = (
        "\nCreating target variable (hypoglycemia prediction) for each patient..."
    )
    MSG_TARGET_HEAD = "\nDataFrame with target variable:"
    MSG_TARGET_COUNTS = (
        "Value counts for target variable (0=No Hypo, 1=Hypo in future):"
    )
    MSG_POSITIVE_PERCENTAGE = "Percentage of positive cases: {:.2f}%"

    # Messages for DataFrame printing
    MSG_DATAFRAME_SAMPLE = "\nDataFrame sample:"
    MSG_DATAFRAME_INFO = "\nDataFrame info:"

    def __init__(self):
        """
        Initialize the DataframeLoader.

        Sets up instance variables for storing DataFrames at different stages
        of the preprocessing pipeline.
        """
        self.raw_df: pd.DataFrame = pd.DataFrame()
        self.resampled_df: pd.DataFrame = pd.DataFrame()
        self.final_df: pd.DataFrame = pd.DataFrame()

    def _load_dataframe(
        self, path: str, num_rows: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Load a dataframe from a CSV file.

        Args:
            path (str): Path to the CSV file
            num_rows (Optional[int]): Number of rows to load. Defaults to DEFAULT_NUM_ROWS.

        Returns:
            pd.DataFrame: Loaded DataFrame with time column converted to datetime
        """
        if num_rows is None:
            num_rows = self.DEFAULT_NUM_ROWS

        df = pd.read_csv(path).head(num_rows)
        df[self.COLUMN_TIME] = pd.to_datetime(df[self.COLUMN_TIME])
        return df

    def _respace_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Resample and impute glucose data for all patients.

        Performs the following operations:
        - Resamples glucose data to fixed frequency (5 minutes)
        - Interpolates missing values using linear interpolation
        - Forward-fills and backward-fills remaining NaNs
        - Drops any remaining NaNs

        Args:
            df (pd.DataFrame): Input DataFrame with patient data

        Returns:
            pd.DataFrame: Processed DataFrame with resampled and imputed data
        """
        # Store processed data for all patients
        processed_data = []

        for patient_id, patient_df in df.groupby(self.COLUMN_ID):
            # Set 'time' as index for resampling
            patient_df = patient_df.set_index(self.COLUMN_TIME).sort_index()

            # Resample to a fixed frequency. 'mean()' aggregates multiple values within a 5min window.
            # In some cases, 'first()' or 'last()' might be more appropriate.
            resampled_df = (
                patient_df[self.COLUMN_GLUCOSE].resample(self.RESAMPLE_FREQ).mean()
            )

            # Interpolate missing values (e.g., if a 5-min slot had no reading or resampling created NaNs)
            # 'linear' interpolation is common for physical measurements like glucose.
            resampled_df = resampled_df.interpolate(method=self.INTERPOLATION_METHOD)

            # Forward-fill any NaNs that might still exist at the beginning of the series (if interpolation can't fill backwards)
            resampled_df = resampled_df.ffill()
            # If still NaNs at beginning (e.g., first few points are NaN), backward-fill
            resampled_df = resampled_df.bfill()

            # Drop any remaining NaNs (e.g., if the entire series was empty or could not be filled)
            resampled_df = resampled_df.dropna()

            if not resampled_df.empty:
                df_temp = resampled_df.to_frame(name=self.COLUMN_GLUCOSE)
                df_temp[self.COLUMN_ID] = patient_id  # Add back the patient ID
                processed_data.append(df_temp)

        # Concatenate all processed patient data
        processed_df = pd.concat(
            processed_data
        ).reset_index()  # 'time' becomes a column again

        return processed_df

    def _identify_hypoglycemic_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create target variable for hypoglycemia prediction.

        Performs the following operations:
        - Identifies hypoglycemic events below threshold
        - Creates rolling window to predict future hypoglycemia
        - Generates binary target variable for prediction

        Args:
            df (pd.DataFrame): Input DataFrame with glucose data

        Returns:
            pd.DataFrame: DataFrame with target variable added
        """
        # Create the target variable per patient
        target_data = []

        logging.info(self.MSG_CREATING_TARGET)

        for patient_id, patient_df in df.groupby(self.COLUMN_ID):
            # Step 1: Identify actual hypoglycemic events in the patient's data
            # Create a boolean series: True where glucose is below threshold
            patient_df[self.COLUMN_IS_HYPO] = (
                patient_df[self.COLUMN_GLUCOSE] < self.HYPO_THRESHOLD
            )

            # Step 2: Determine if a future hypoglycemic event occurs within the horizon
            # 'rolling' window, centered=False to look into the future
            # The window size needs to correspond to the PREDICTION_HORIZON_MINUTES
            # Window size in terms of number of samples (e.g., 60 minutes / 5 minutes per sample = 12 samples)
            window_samples = self.PREDICTION_HORIZON_MINUTES // (
                pd.to_timedelta(self.RESAMPLE_FREQ).total_seconds() / 60
            )

            # To predict hypoglycemia *in the future*, we use a 'rolling' window that comes *after* the current point.
            # A simple way to achieve this is to reverse the series, apply a 'rolling max' (to check for *any* True in past window),
            # and then reverse back. This indicates if any *future* point (within the horizon) is hypoglycemic.
            # We use 'max' on the boolean series because True=1, False=0. Max will be 1 if any True exists.
            patient_df[self.COLUMN_IS_HYPO_EVENT_IN_FUTURE] = (
                patient_df[self.COLUMN_IS_HYPO]
                .rolling(
                    window=int(window_samples),
                    min_periods=self.ROLLING_MIN_PERIODS,  # Allow smaller windows at the start/end
                    closed=self.ROLLING_CLOSED_PARAM,  # Ensure the current point is included if needed for future horizon
                )
                .max()
                .shift(-int(window_samples) + 1)
            )  # Shift back to align with the current time point

            # Fill NaNs created by shifting at the end of the series (no future data)
            patient_df[self.COLUMN_IS_HYPO_EVENT_IN_FUTURE] = (
                patient_df[self.COLUMN_IS_HYPO_EVENT_IN_FUTURE]
                .fillna(self.FILL_VALUE)
                .astype(self.DATA_TYPE_INT)
            )

            # The actual observation 'is_hypo' is not our direct target, but 'is_hypo_event_in_future' is.
            # Drop the intermediate 'is_hypo' column
            patient_df = patient_df.drop(columns=[self.COLUMN_IS_HYPO])

            target_data.append(patient_df)

        final_df = pd.concat(target_data).reset_index(drop=True)
        return final_df

    def load_dataframe(
        self, csv_path: str, num_rows: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Execute the complete data preprocessing pipeline.

        Performs the following steps:
        1. Load CSV data with time conversion
        2. Resample and impute glucose data
        3. Create target variables for hypoglycemia prediction

        Args:
            csv_path (str): Path to the CSV file to load
            num_rows (Optional[int]): Number of rows to load. Defaults to DEFAULT_NUM_ROWS.

        Returns:
            pd.DataFrame: Final processed DataFrame ready for model training
        """
        # Step 1: Load data
        self.raw_df = self._load_dataframe(csv_path, num_rows)

        # Step 2: Respace data
        self.resampled_df = self._respace_data(self.raw_df)

        # Step 3: Create target variables
        self.final_df = self._identify_hypoglycemic_events(self.resampled_df)

        return self.final_df
