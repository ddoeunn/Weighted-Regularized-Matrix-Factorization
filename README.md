# Weighted-Regularized-Matrix-Factorization
Weighted Regularized Matrix Factorization for Implicit Feedback in Recommender System


* [[1]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=4781121&casa_token=fFLKRi3Rik4AAAAA:U2wCKtmX8KkdOe86FE08TZO4i8rTnxW0-WCw5ydvR01FqNVNTIjbH4YZmBzQzwdIQ9MTNouPWuc&tag=1) Hu, Yifan, Yehuda Koren, and Chris Volinsky. "Collaborative filtering for implicit feedback datasets." 2008.
* [[2]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=4781145&casa_token=mV7FiNkLbIwAAAAA:0S5KcW0Rjrw-kKq3DLChQlHUnjtm8xuFK9izYUGpZSbFK_f2oh8Q7wNvBmwX8jctDzs-TnEYpbE) Pan, Rong, et al. "One-class collaborative filtering." 2008.
* [[3]](https://dl.acm.org/doi/pdf/10.1145/2911451.2911489?casa_token=ppDtD4EAfpMAAAAA:YhRqsYPdd5jWt-zOSkIimM6-AYn2pGxzARZlqTlf0SM-Qi8e7B0h5AbdbDLlIIWnRu454rr-o4YGpok) He, Xiangnan, et al. "Fast matrix factorization for online recommendation with implicit feedback."  2016.

I implemented WRMF methods in python in reference to [Cornac](https://cornac.readthedocs.io/en/latest/). Only uniform weighting strategy on positive or negative instances is available in WMF model in Cornac. By modifying the WMF code of Cornac,  I implemented user-oriented, item-oriented weighting strategy of "One-class collaborative filtering (Pan, Rong, et al.)" and item-popularity weighting strategy of "Fast matrix factorization for online recommendation with implicit feedback (He, Xiangnan, et al)".

See the [example](https://github.com/ddoeunn/Weighted-Regularized-Matrix-Factorization/blob/main/Example/Example.ipynb) comparing weighting strategies.

|      Data     	|     Strategy    	|  k 	| Train time (s) 	| Precision@k 	| Recall@k 	|  NDCG@k  	|
|:-------------:	|:---------------:	|:--:	|:-------:	|:-----------:	|:--------:	|:--------:	|
| movielens100k 	|   uniform_pos   	| 10 	|  6.2740 	|   0.292365  	| 0.184272 	| 0.343978 	|
| movielens100k 	|   uniform_neg   	| 10 	|  9.2785 	|   0.327253  	| 0.215326 	| 0.383740 	|
| movielens100k 	|  user_oriented  	| 10 	| 10.6478 	|   0.366172  	| 0.230124 	| 0.431030 	|
| movielens100k 	|  item_oriented  	| 10 	|  9.8481 	|   0.361082  	| 0.229981 	| 0.426998 	|
| movielens100k 	| item_popularity 	| 10 	| 11.0064 	|   0.360551  	| 0.231452 	| 0.423511 	|



```{.python}
from Datasets import Movielens
from Evaluation.data_split import split_data
from Evaluation.ranking_metrics import *
from WRMF.wrmf import *
from WRMF import wrmf_rec

df_movielens = Movielens.load_data()                # load dataset
train, test = split_data(df_movielens,
                         split_strategy="random_by_user",
                         random_state=0)            # split data

wrmf = WRMF(train, weight_strategy="uniform_pos")   # wrmf model
model =  train_cornac(wrmf, train)                  # train model

k = 10
top_k = wrmf_rec.recommend_top_k(model, train, k)   # recommendation
ranking_metrics(top_k, test)                        # evaluation

```

---
## **Weighted Regularized Matrix Factorization (WRMF)**

Basic idea of Weighted Regularized Matrix Factorization (WRMF) is to assign smaller weights to the unobserved instances than the observed. The weights are related to the concept of confidence. As not interacting with an item can result from other reasons than not liking it, negative instances have low confidence. For example, a user might be unaware of the existence of the item, or unable to consume it due to its price or limited availability. Unobserved instances are a mixture of negative and unknown feedback.

 Also, interacting with an item can be caused by a variety of reasons that differ from liking it. For example, a user may buy an item as gift for someone else, despite the user does not like the item. Thus it can be thought that there are also different confidence levels among the items that the user interacted with.

Several weighting strategies have been proposed. Read more [here!](https://ddoeunn.github.io/2021/05/02/SUMMARY-Weighted-Matrix-Factorization-for-Implicit-Feedback.md.html) for more details.
