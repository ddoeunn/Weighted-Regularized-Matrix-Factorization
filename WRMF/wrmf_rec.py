import pandas as pd
import cornac
from WRMF.wrmf_utils import *
from utils.common.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_PREDICTION_COL,
)


def predict_score(
        model,
        data,
        user_col=DEFAULT_USER_COL,
        item_col=DEFAULT_ITEM_COL,
        pred_col=DEFAULT_PREDICTION_COL,
        remove_seen=True,
):
    """Computes predictions of recommender model from Cornac on all users and items in data.
    It can be used for computing ranking metrics like NDCG.
    original source code - "https://github.com/microsoft/recommenders/blob/main/reco_utils/recommender/cornac/cornac_utils.py"
    Args:
        model (cornac.models.Recommender): a recommender model from Cornac
        data (pd.DataFrame): the data from which to get the users and items
        user_col (str): name of the user column
        item_col (str): name of the item column
        pred_col (str): name of the prediction column
        remove_seen (bool): flag to remove (user, item) pairs seen in the training data
    Returns:
        pd.DataFrame: dataframe with usercol, itemcol, predcol
    """
    users, items, preds = [], [], []
    item = list(model.train_set.iid_map.keys())
    for uid, user_idx in model.train_set.uid_map.items():
        user = [uid] * len(item)
        users.extend(user)
        items.extend(item)
        preds.extend(model.score(user_idx).tolist())

    all_predictions = pd.DataFrame(
        data={user_col: users, item_col: items, pred_col: preds}
    )

    if remove_seen:
        df_tmp = pd.concat(
            [
                data[[user_col, item_col]],
                pd.DataFrame(
                    data=np.ones(data.shape[0]), columns=["dummycol"], index=data.index
                ),
            ],
            axis=1,
        )
        merged = pd.merge(df_tmp, all_predictions, on=[user_col, item_col], how="outer")
        return merged[merged["dummycol"].isnull()].drop("dummycol", axis=1)
    else:
        return all_predictions


def recommend_top_k(model, data, k):
    """Recommend top-k items for each users

    Args:
        model (cornac.models.Recommender): a recommender model from Cornac
        data (pd.DataFrame): the data from which to get the users and items
        k (int): top-k for recommendation

    Returns:

    """
    df_pred_score = predict_score(model, data)
    top_k_items = (
        df_pred_score.groupby(DEFAULT_USER_COL, as_index=False).apply(
            lambda x: x.nlargest(k, DEFAULT_PREDICTION_COL)).reset_index(drop=True)
    )
    top_k_items["rank"] = top_k_items.groupby(DEFAULT_USER_COL, sort=False).cumcount() + 1
    top_k_recommend = top_k_items.pivot_table(values=DEFAULT_ITEM_COL, index=DEFAULT_USER_COL,
                                              columns='rank')

    return top_k_recommend