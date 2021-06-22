import pandas as pd
from utils.common.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
)
from Datasets.download_utils import (
    download_path, maybe_download, extract_file_from_zip
)

URL_LAST_FM = "http://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip"
LAST_FM_FILE_NAME = "user_artists.dat"


def load_data(header=None, local_cache_path=None, unzip_path=None):
    """ Load Last.FM Dataset
    Download the dataset from https://grouplens.org/datasets/hetrec-2011/
    Notice that this data doesn't contain timestamp column.

    Args:
        header (list or tuple or None): dataset header.
        local_cache_path (str or None): Path to cache the downloaded file.
            If None, all the intermediate files will be stored in a temporary directory and removed after use.
        unzip_path (str): Path to save extracted file from zip file.
    Returns:
        pd.DataFrame: Last.FM Datset
    """

    if header is None:
        header = [DEFAULT_USER_COL, DEFAULT_ITEM_COL, DEFAULT_RATING_COL]

    with download_path(local_cache_path) as path:
        zip_file_path = maybe_download(URL_LAST_FM, work_directory=path)
        data_file_path = extract_file_from_zip(zip_file_path, LAST_FM_FILE_NAME, unzip_path)
        last_fm_df = pd.read_table(data_file_path, header=None, sep=r"\s", skiprows=[0],
                                   names=header, engine='python')

    return last_fm_df
