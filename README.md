# Weighted-Regularized-Matrix-Factorization
Weighted Regularized Matrix Factorization for Implicit Feedback in Recommender System


* [[1]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=4781121&casa_token=fFLKRi3Rik4AAAAA:U2wCKtmX8KkdOe86FE08TZO4i8rTnxW0-WCw5ydvR01FqNVNTIjbH4YZmBzQzwdIQ9MTNouPWuc&tag=1) Hu, Yifan, Yehuda Koren, and Chris Volinsky. "Collaborative filtering for implicit feedback datasets." 2008.
* [[2]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=4781145&casa_token=mV7FiNkLbIwAAAAA:0S5KcW0Rjrw-kKq3DLChQlHUnjtm8xuFK9izYUGpZSbFK_f2oh8Q7wNvBmwX8jctDzs-TnEYpbE) Pan, Rong, et al. "One-class collaborative filtering." 2008.
* [[3]](https://dl.acm.org/doi/pdf/10.1145/2911451.2911489?casa_token=ppDtD4EAfpMAAAAA:YhRqsYPdd5jWt-zOSkIimM6-AYn2pGxzARZlqTlf0SM-Qi8e7B0h5AbdbDLlIIWnRu454rr-o4YGpok) He, Xiangnan, et al. "Fast matrix factorization for online recommendation with implicit feedback."  2016.

I implemented WRMF methods in python in reference to [Cornac](https://cornac.readthedocs.io/en/latest/). Only uniform weighting strategy on positive or negative instances is available in WMF model in Cornac. By modifying the WMF code of Cornac,  I implemented user-oriented, item-oriented weighting strategy of "One-class collaborative filtering (Pan, Rong, et al.)" and item-popularity weighting strategy of "Fast matrix factorization for online recommendation with implicit feedback (He, Xiangnan, et al)".



---
## **Weighted Regularized Matrix Factorization (WRMF)**  

Basic idea of Weighted Regularized Matrix Factorization (WRMF) is to assign smaller weights to the unobserved instances than the observed. The weights are related to the concept of confidence. As not interacting with an item can result from other reasons than not liking it, negative instances have low confidence. For example, a user might be unaware of the existence of the item, or unable to consume it due to its price or limited availability. Unobserved instances are a mixture of negative and unknown feedback.

 Also, interacting with an item can be caused by a variety of reasons that differ from liking it. For example, a user may buy an item as gift for someone else, despite the user does not like the item. Thus it can be thought that there are also different confidence levels among the items that the user interacted with.

Several weighting strategies have been proposed. Read more [here]() for more details.
