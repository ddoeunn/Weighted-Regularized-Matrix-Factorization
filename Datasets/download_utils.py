import os
import logging
import math
from tqdm import tqdm
import shutil
import zipfile
from tempfile import TemporaryDirectory
from contextlib import contextmanager
import requests

log = logging.getLogger(__name__)


def maybe_download(url, filename=None, work_directory=".", expected_bytes=None):
    """Download a file if it is not already downloaded.
    original source code - "https://github.com/microsoft/recommenders/blob/main/reco_utils/dataset/download_utils.py"

    Args:
        url (str): URL of the file to download.
        filename (str): File name.
        work_directory (str): Working directory.
        expected_bytes (int): Expected file size in bytes.

    Returns:
         str: File path of the file downloaded.
    """

    if filename is None:
        filename = url.split("/")[-1]
    os.makedirs(work_directory, exist_ok=True)
    filepath = os.path.join(work_directory, filename)

    if not os.path.exists(filepath):
        r = requests.get(url, stream=True)
        total_size = int(r.headers.get("content-length", 0))
        block_size = 1024
        num_iterables = math.ceil(total_size / block_size)

        with open(filepath, "wb") as file:
            for data in tqdm(
                    r.iter_content(block_size),
                    total=num_iterables,
                    unit="KB",
                    unit_scale=True,
            ):
                file.write(data)
    else:
        log.info("File {} already downloaded".format(filepath))
    if expected_bytes is not None:
        statinfo = os.stat(filepath)
        if statinfo.st_size != expected_bytes:
            os.remove(filepath)
            raise IOError("Failed to verify {}".format(filepath))

    return filepath


@contextmanager
def download_path(path=None):
    """Return a path to download data. If `path=None`, then it yields a temporal path that is eventually deleted,
    otherwise the real path of the input.
    original source code - "https://github.com/microsoft/recommenders/blob/main/reco_utils/dataset/download_utils.py"

    Args:
        path (str): Path to download data.
    Returns:
        str: Real path where the data is stored.

    """

    if path is None:
        tmp_dir = TemporaryDirectory()
        try:
            yield tmp_dir.name
        finally:
            tmp_dir.cleanup()
    else:
        path = os.path.realpath(path)
        yield path


def extract_file_from_zip(zip_path, file_path, path=None):
    """extract file from zip file

    Args:
        zip_path (str): Zip file path
        file_path (str): Path with files to extract
        path (str): Path to save extracted file

    Returns:
        str: File path of the file extracted.
    """

    _, file_name = os.path.split(file_path)
    dirs, folder_name = os.path.split(zip_path)

    if path is None:
        folder_path = os.path.join(dirs, folder_name.split('.')[0])
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
        extracted_path = os.path.join(folder_path, file_name)

    else:
        extracted_path = os.path.join(path, file_name)

    with zipfile.ZipFile(zip_path, "r") as z:
        with z.open(file_path) as zf, open(extracted_path, "wb") as f:
            shutil.copyfileobj(zf, f)

    return extracted_path



