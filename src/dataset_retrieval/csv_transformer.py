import pandas as pd
import os
import logging
from typing import Tuple
from sas7bdat import SAS7BDAT


class CsvTransformer:
    """
    Class for transforming SAS dataset files to CSV format.

    This class handles the complete pipeline of reading SAS files, merging datasets,
    processing data, and saving to CSV format. It includes data cleaning, column
    renaming, and patient ID anonymization.
    """

    # === CONSTANTS ===
    AGE = "AgeAsofEnrollDt"
    AGE_COL = "age"
    CGM_FILE = "iobp2devicecgm.sas7bdat"
    DATASET_DIR = "dataset"
    DEVICE_TIME = "DeviceDtTm"
    GL = "gl"
    ID = "id"
    INSULIN_MOD = "InsModPump"
    INSULIN_MODALITY = "insulinModality"
    JOIN_TYPE = "left"
    OUTPUT_DIR = "csv_data"
    OUTPUT_FILE = "lynch2022.csv"
    PATIENT_ID = "PtID"
    PSEUDO_ID = "pseudoID"
    PSEUDO_ID_OFFSET = 1000
    ROSTER_FILE = "iobp2ptroster.sas7bdat"
    SAS_DIR = "Data Tables in SAS"
    SCREENING_FILE = "iobp2diabscreening.sas7bdat"
    SEX = "Sex"
    SEX_COL = "sex"
    TIME = "time"
    TIME_FORMAT = "%m/%d/%Y %I:%M:%S %p"
    VALUE = "Value"

    # Messages
    MSG_COMPLETED = "🎯 Processing completed successfully!"
    MSG_FILES_LOADED = "✅ SAS files loaded successfully!"
    MSG_READING_FILES = "📁 Reading SAS files..."
    MSG_SAVED_TO = "✅ Saved to: {path}"
    MSG_TITLE = "🔬 IOBP2 Dataset Preprocessor"
    MSG_WRITING_CSV = "💾 Writing to CSV..."

    def __init__(self):
        """
        Initialize the CsvTransformer.

        Sets up instance variables for storing DataFrames at different stages
        of the transformation process.
        """
        # Initialize with empty DataFrames to satisfy type checker
        self.data = pd.DataFrame()
        self.demo = pd.DataFrame()
        self.age = pd.DataFrame()
        self.merged_data = pd.DataFrame()
        self.processed_df = pd.DataFrame()

    def _load_sas_file(self, file_path: str) -> pd.DataFrame:
        """
        Load a single SAS file and return as DataFrame.

        Args:
            file_path (str): Path to the SAS file to load

        Returns:
            pd.DataFrame: The loaded SAS data as a pandas DataFrame
        """
        with SAS7BDAT(file_path) as sas_file:
            return sas_file.to_data_frame()

    def _read_sas_files(self) -> None:
        """
        Read all required SAS files and store as instance variables.

        Loads three SAS files: CGM data, demographic data, and age data.
        Stores them in self.data, self.demo, and self.age respectively.
        """
        logging.info(self.MSG_READING_FILES)
        sas_directory = os.path.join(self.DATASET_DIR, self.SAS_DIR)
        self.data = self._load_sas_file(os.path.join(sas_directory, self.CGM_FILE))
        self.demo = self._load_sas_file(
            os.path.join(sas_directory, self.SCREENING_FILE)
        )
        self.age = self._load_sas_file(os.path.join(sas_directory, self.ROSTER_FILE))
        logging.info(self.MSG_FILES_LOADED)

    def _select_columns(
        self, data: pd.DataFrame, demo: pd.DataFrame, age: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Select only the necessary columns from each DataFrame.

        Args:
            data (pd.DataFrame): CGM data with patient ID, device time, and glucose values
            demo (pd.DataFrame): Demographic data with patient ID, insulin modality, and sex
            age (pd.DataFrame): Age data with patient ID and age information

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Selected columns from each DataFrame
        """
        data_selected = data[[self.PATIENT_ID, self.DEVICE_TIME, self.VALUE]]
        demo_selected = demo[[self.PATIENT_ID, self.INSULIN_MOD, self.SEX]]
        age_selected = age[[self.PATIENT_ID, self.AGE]]
        return (data_selected, demo_selected, age_selected)  # type: ignore

    def _merge_datasets(self) -> None:
        """
        Merge the three datasets on patient ID.

        Performs left joins to combine CGM data with demographic and age information
        based on patient ID. Stores the merged result in self.merged_data.
        """
        data_selected, demo_selected, age_selected = self._select_columns(
            self.data, self.demo, self.age
        )
        merged_data = data_selected.merge(
            demo_selected, on=self.PATIENT_ID, how=self.JOIN_TYPE
        )
        self.merged_data = merged_data.merge(
            age_selected, on=self.PATIENT_ID, how=self.JOIN_TYPE
        )

    def _process_columns_and_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rename columns, process data types, and clean the dataset.

        Performs the following operations:
        - Renames columns to standardized format
        - Converts time column to datetime format
        - Processes insulin modality (0=injections, 1=pump)
        - Removes rows with missing time or glucose values
        - Sorts by patient ID and time
        - Creates anonymized patient IDs
        - Selects final columns for output

        Args:
            df (pd.DataFrame): Input DataFrame to process

        Returns:
            pd.DataFrame: Processed and cleaned DataFrame
        """
        # Rename columns to final format
        column_mapping = {
            self.PATIENT_ID: self.ID,
            self.DEVICE_TIME: self.TIME,
            self.VALUE: self.GL,
            self.AGE: self.AGE_COL,
            self.SEX: self.SEX_COL,
            self.INSULIN_MOD: self.INSULIN_MODALITY,
        }
        df = df.rename(columns=column_mapping)

        # Convert time column to datetime
        df[self.TIME] = pd.to_datetime(
            df[self.TIME], format=self.TIME_FORMAT, errors="coerce"
        )

        # Set insulin modality to 0 for insulin injections, 1 for insulin pump
        df[self.INSULIN_MODALITY] = df[self.INSULIN_MODALITY].fillna(0).astype(int)

        # Remove rows with missing time or glucose values
        df = df.dropna(subset=[self.TIME, self.GL])

        # Sort by ID and time, then create pseudo IDs for patients
        df = df.sort_values([self.ID, self.TIME])
        df[self.PSEUDO_ID] = df.groupby(self.ID).ngroup() + self.PSEUDO_ID_OFFSET

        # Select final columns and rename pseudo ID back to ID
        selected_columns = [
            self.PSEUDO_ID,
            self.TIME,
            self.GL,
            self.AGE_COL,
            self.SEX_COL,
            self.INSULIN_MODALITY,
        ]
        df = df[selected_columns]  # type: ignore
        df = df.rename(columns={self.PSEUDO_ID: self.ID})

        return df

    def _process_data(self) -> None:
        """
        Process and clean the merged data.

        Applies data processing pipeline to the merged dataset and stores
        the final processed DataFrame in self.processed_df. Prints summary
        statistics of the processed data.
        """
        df = self.merged_data.copy()
        self.processed_df = self._process_columns_and_data(df)

    def _save_to_csv(self) -> None:
        """
        Save the processed DataFrame to CSV file.

        Creates the output directory if it doesn't exist and saves the
        processed DataFrame to the specified CSV file. Prints confirmation
        messages with file path and final data shape.
        """
        if not os.path.exists(self.OUTPUT_DIR):
            os.makedirs(self.OUTPUT_DIR)
        output_path = os.path.join(self.OUTPUT_DIR, self.OUTPUT_FILE)
        self.processed_df.to_csv(output_path, index=False)
        logging.info(self.MSG_SAVED_TO.format(path=output_path))

    def _generate_dataframe(self) -> None:
        """
        Generate the final processed DataFrame.

        Executes the complete data transformation pipeline:
        1. Read SAS files
        2. Merge datasets
        3. Process and clean data
        """
        self._read_sas_files()
        self._merge_datasets()
        self._process_data()

    def generate_dataframe(self) -> pd.DataFrame:
        """
        Main execution method that transforms SAS files to CSV.

        Orchestrates the complete transformation pipeline from SAS files
        to CSV format. This is the primary entry point for the class.

        Returns:
            pd.DataFrame: The final processed DataFrame that was saved to CSV
        """
        logging.info(self.MSG_TITLE)
        self._generate_dataframe()
        self._save_to_csv()
        logging.info(self.MSG_COMPLETED)
        return self.processed_df
