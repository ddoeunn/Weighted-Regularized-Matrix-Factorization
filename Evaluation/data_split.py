import math
from tqdm import tqdm
from tabulate import tabulate
from sklearn.utils import shuffle
from utils.common.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_TIMESTAMP_COL,
    DEFAULT_SPLIT_FLAG,
    DEFAULT_TEST_SIZE,
    DEFAULT_VAL_SIZE
)

SPLIT_STRATEGY = ("leave_one_last", "temporal_user", "temporal_global", "random_by_user")
TIMESTAMP_STRATEGY = ("leave_one_last", "temporal_user", "temporal_global")
ERROR_SPLIT_STRATEGY = "Invalid split strategy. " \
                       "Should be one of {'leave_one_last', 'temporal_user', 'temporal_global', 'random_by_user'}"
ERROR_TIMESTAMP_COL = ("This split strategy is available only for data with timestamp columns.\n"
                       "There is no timestamp column in this data. Only 'random_by_user' is available.\n")


def check_no_timestamp_col(data):
    return DEFAULT_TIMESTAMP_COL not in data.columns


def get_intersect(train, data_to_remove):
    """Remove user and item ids not in train from test/val.

    Args:
        train (DataFrame): train data.
        data_to_remove (list): list of data to remove. [test] or [test, val]

    Returns:

    """
    user_pool = list(train[DEFAULT_USER_COL].unique())
    item_pool = list(train[DEFAULT_ITEM_COL].unique())

    for data in data_to_remove:
        data.drop(
            data[~data[DEFAULT_USER_COL].isin(user_pool)].index,
            inplace=True,
        )
        data.drop(
            data[~data[DEFAULT_ITEM_COL].isin(item_pool)].index,
            inplace=True,
        )


def leave_one_last(data, validation=False):
    """Leave one last split

    Args:
        data (DataFrame): data to split.
        validation (bool): Default False. If True, split train/val/test. If False, split train/test.

    Returns:
        DataFrame: split labeled DataFrame (split_flag = train/val/test)
    """

    if check_no_timestamp_col(data):
        print(ERROR_TIMESTAMP_COL)
        return None

    df_split = data.copy()
    df_split[DEFAULT_SPLIT_FLAG] = 'train'
    df_split.sort_values(by=[DEFAULT_TIMESTAMP_COL], ascending=False, inplace=True)

    if validation:
        df_split.loc[df_split.groupby([DEFAULT_USER_COL]).head(2).index, DEFAULT_SPLIT_FLAG] = 'val'

    df_split.loc[df_split.groupby([DEFAULT_USER_COL]).head(1).index, DEFAULT_SPLIT_FLAG] = 'test'
    df_split = df_split.sort_values([DEFAULT_USER_COL, DEFAULT_ITEM_COL]).reset_index(drop=True)

    return df_split


def temporal_user_split(data, test_size=DEFAULT_TEST_SIZE, val_size=DEFAULT_VAL_SIZE,
                        validation=False):
    """Temporal user split

    Args:
        data (DataFrame): data to split.
        test_size (float): the proportion of the dataset to include in the test split.
        val_size (float): the proportion of the dataset to include in the val split.
        validation (bool): Default False. If True, split train/val/test. If False, split train/test.

    Returns:
        DataFrame: split labeled DataFrame (split_flag = train/val/test)
    """

    if check_no_timestamp_col(data):
        print(ERROR_TIMESTAMP_COL)
        return None

    users = data[DEFAULT_USER_COL].unique()
    df_split = data.copy()
    df_split[DEFAULT_SPLIT_FLAG] = 'train'
    df_split.sort_values(by=[DEFAULT_TIMESTAMP_COL], ascending=True, inplace=True)

    for u in tqdm(users):
        user_index = df_split[df_split[DEFAULT_USER_COL] == u].index.values
        user_size = len(user_index)
        test_split_size = math.ceil(user_size * test_size)
        train_size = user_size - test_split_size

        df_split.loc[user_index[train_size:], DEFAULT_SPLIT_FLAG] = 'test'

        if validation:
            val_split_size = math.ceil(user_size * val_size)
            df_split.loc[
                user_index[(train_size - val_split_size): train_size], DEFAULT_SPLIT_FLAG
            ] = 'val'

    df_split = df_split.sort_values([DEFAULT_USER_COL, DEFAULT_ITEM_COL]).reset_index(drop=True)

    return df_split


def temporal_global_split(data, test_size=DEFAULT_TEST_SIZE, val_size=DEFAULT_VAL_SIZE,
                          validation=False):
    """Temporal global split

    Args:
        data (DataFrame): data to split.
        test_size (float): the proportion of the dataset to include in the test split.
        val_size (float): the proportion of the dataset to include in the val split.
        validation (bool): Default False. If True, split train/val/test. If False, split train/test.

    Returns:
        DataFrame: split labeled DataFrame (split_flag = train/val/test)
    """

    if check_no_timestamp_col(data):
        print(ERROR_TIMESTAMP_COL)
        return None

    df_split = data.copy()
    df_split[DEFAULT_SPLIT_FLAG] = 'train'
    df_split.sort_values(by=[DEFAULT_TIMESTAMP_COL], ascending=True, inplace=True)

    data_index = df_split.index.values
    data_size = len(data_index)
    test_split_size = math.ceil(data_size * test_size)
    train_size = data_size - test_split_size

    df_split.loc[data_index[train_size:], DEFAULT_SPLIT_FLAG] = 'test'

    if validation:
        val_split_size = math.ceil(data_size * val_size)
        df_split.loc[
            data_index[(train_size - val_split_size): train_size], DEFAULT_SPLIT_FLAG
        ] = 'val'

    df_split = df_split.sort_values([DEFAULT_USER_COL, DEFAULT_ITEM_COL]).reset_index(drop=True)

    return df_split


def random_split_by_user(data, test_size=DEFAULT_TEST_SIZE, val_size=DEFAULT_VAL_SIZE,
                         validation=False, random_state=None):
    """Random split by user

    Args:
        data (DataFrame): data to split.
        test_size (float): the proportion of the dataset to include in the test split.
        val_size (float): the proportion of the dataset to include in the val split.
        validation (bool): Default False. If True, split train/val/test. If False, split train/test.
        random_state (int): random seed.

    Returns:
        DataFrame: split labeled DataFrame (split_flag = train/val/test)
    """

    users = data[DEFAULT_USER_COL].unique()
    df_split = data.copy()
    df_split[DEFAULT_SPLIT_FLAG] = 'train'

    for u in tqdm(users):
        user_index = df_split[df_split[DEFAULT_USER_COL] == u].index.values
        user_index = shuffle(user_index, random_state=random_state)

        user_size = len(user_index)
        test_split_size = math.ceil(user_size * test_size)
        train_size = user_size - test_split_size

        df_split.loc[user_index[train_size:], DEFAULT_SPLIT_FLAG] = 'test'

        if validation:
            val_split_size = math.ceil(user_size * val_size)
            df_split.loc[
                user_index[(train_size - val_split_size):train_size], DEFAULT_SPLIT_FLAG
            ] = 'val'

    df_split = df_split.sort_values([DEFAULT_USER_COL, DEFAULT_ITEM_COL]).reset_index(drop=True)

    return df_split


def split_data(data, split_strategy='leave_one_last',
               test_size=DEFAULT_TEST_SIZE, val_size=DEFAULT_VAL_SIZE,
               validation=False, random_state=None, intersect=True):
    """split data into train/val/test or train/test based on split strategy.

    Args:
        data (DataFrame): data to split
        split_strategy (str): Data split strategy
                                - "leave_one_last", "temporal_user", "temporal_global", "random_by_user"
        test_size (float): the proportion of the dataset to include in the test split
        val_size (float): the proportion of the dataset to include in the val split
        validation (bool): Default False. If True, split train/val/test. If False, split train/test
        random_state (int): random state
        intersect (bool): if True, return split data which is removed user and item ids not in train.

    Returns:
        DataFrame : If intersect=True, return (train, test) or (train, val, test).
                    If False, return split dataFrame with labeled split_flag
    """

    if split_strategy == "leave_one_last":
        df_split = leave_one_last(data, validation)
    elif split_strategy == "temporal_user":
        df_split = temporal_user_split(data, test_size, val_size, validation)
    elif split_strategy == "temporal_global":
        df_split = temporal_global_split(data, test_size, val_size, validation)
    elif split_strategy == "random_by_user":
        df_split = random_split_by_user(data, test_size, val_size, validation, random_state)
    else:
        print(ERROR_SPLIT_STRATEGY)
        return None

    if df_split is None:
        return None

    if not intersect:
        return df_split

    train = df_split[df_split[DEFAULT_SPLIT_FLAG] == 'train'].drop(
        DEFAULT_SPLIT_FLAG,
        axis=1
    )
    test = df_split[df_split[DEFAULT_SPLIT_FLAG] == 'test'].drop(
        DEFAULT_SPLIT_FLAG,
        axis=1
    )

    if validation:
        val = df_split[df_split[DEFAULT_SPLIT_FLAG] == 'val'].drop(
            DEFAULT_SPLIT_FLAG,
            axis=1
        )
        if intersect:
            get_intersect(train, [test, val])
        return train, val, test

    else:
        if intersect:
            get_intersect(train, [test])
        return train, test





