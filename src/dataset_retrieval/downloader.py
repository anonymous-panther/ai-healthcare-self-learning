import urllib.request
import urllib.error
import os
import zipfile
import logging
from typing import Tuple


class DatasetDownloader:
    """
    Main class for handling dataset download operations.

    This class handles direct download of the Insulin-Only Bionic Pancreas
    Pivotal (IOBP2) dataset from the public S3 URL.
    """

    # === CONSTANTS ===
    DATASET_URL = "https://live-jchrpublicdatasets.s3.amazonaws.com/Diabetes/Public%20Datasets/IOBP2%20RCT%20Public%20Dataset.zip"
    OUTPUT_ZIP_FILE = "downloaded_dataset.zip"
    UNZIP_DIRECTORY = "dataset"
    CHUNK_SIZE = 8192

    # Messages
    MSG_TITLE = (
        "=== Insulin-Only Bionic Pancreas Pivotal Dataset Download Initiated ==="
    )
    MSG_DOWNLOAD_ERROR = "Error downloading file: {error}"
    MSG_EXTRACTION_ERROR = "Error extracting file: {error}"
    MSG_EXTRACTION_FAILED = "❌ Failed to extract the dataset file from: {path}"
    MSG_DOWNLOAD_FAILED = "❌ Failed to download the dataset file"

    def __init__(self):
        """Initialize the downloader."""
        self.zip_path: str = ""
        self.extract_path: str = ""

    def _download_file(self, file_url: str, output_path: str) -> bool:
        """
        Download a file from URL and save to local path.

        Args:
            file_url: URL of the file to download
            output_path: Local path where to save the file

        Returns:
            True if download successful, False otherwise
        """
        try:
            # Create output directory if needed
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            # Use urllib to download the file with timeout
            with urllib.request.urlopen(file_url, timeout=60) as response:
                file_size = int(response.headers.get("Content-Length", 0))
                downloaded = 0

                with open(output_path, "wb") as f:
                    while True:
                        chunk = response.read(self.CHUNK_SIZE)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)

                        # Show progress if file size is known
                        if file_size > 0:
                            progress = (downloaded / file_size) * 100
                            if (
                                downloaded % (self.CHUNK_SIZE * 100) == 0
                            ):  # Show every 100 chunks
                                logging.info(f"Download progress: {progress:.1f}%")

            return True

        except urllib.error.URLError as e:
            logging.error(f"URL error: {e}")
            return False
        except Exception as e:
            logging.error(self.MSG_DOWNLOAD_ERROR.format(error=e))
            return False

    def _extract_zip(self, zip_path: str, extract_to: str) -> bool:
        """
        Extract a zip file to the specified directory.

        Args:
            zip_path: Path to the zip file
            extract_to: Directory where to extract files

        Returns:
            True if extraction successful, False otherwise
        """
        try:

            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_to)

            return True

        except Exception as e:
            logging.error(self.MSG_EXTRACTION_ERROR.format(error=e))
            return False

    def _process_download(self) -> Tuple[bool, str, str]:
        """
        Complete download and extraction process.

        Returns:
            Tuple of (success, zip_path, extract_path)
        """
        output_path = self.OUTPUT_ZIP_FILE
        success = self._download_file(self.DATASET_URL, output_path)

        if not success:
            logging.error(self.MSG_DOWNLOAD_FAILED)
            return False, "", ""

        unzip_dir = self.UNZIP_DIRECTORY
        os.makedirs(unzip_dir, exist_ok=True)

        extract_success = self._extract_zip(output_path, unzip_dir)

        if extract_success:
            return True, output_path, unzip_dir
        else:
            logging.error(self.MSG_EXTRACTION_FAILED.format(path=output_path))
            return False, output_path, unzip_dir

    def get_dataset(self) -> Tuple[bool, str, str]:
        """
        Main execution method that handles the complete download process.

        Returns:
            Tuple of (success, zip_path, extract_path)
        """
        logging.info(self.MSG_TITLE)

        success, zip_path, extract_path = self._process_download()
        self.zip_path = zip_path
        self.extract_path = extract_path

        return success, zip_path, extract_path
