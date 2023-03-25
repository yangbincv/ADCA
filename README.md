# Augmented Dual-Contrastive Aggregation Learning for Unsupervised Visible-Infrared Person Re-Identification ACM MM22

# Highlight

1. We propose a dual-stream contrastive learning framework with two modality-specific memory modules for USL-VI-ReID. To learn color-invariant features, the visible stream employs a powerful color augmentation method of random channel augmentation as a bridge to infrared modality for joint contrastive learning.
2. We design a Cross-modality Memory Aggregation (CMA) module to select reliable positive samples and aggregate corresponding memory representations in a parameter-free manner, which enables the dual-stream framework to learn better modality-invariant knowledge, while simultaneously reinforcing each contrastive learning stream.
3. We present extensive experiments on the SYSU-MM01 and RegDB datasets, which demonstrate that our method outperforms existing unsupervised methods under various settings, and even surpasses some supervised counterparts, providing a new baseline for USL-VI-ReID task and significantly pushing VI-ReID to real-world deployment.

# Dataset
Put SYSU-MM01 and RegDB dataset into data/sysu and data/regdb, run prepare\_sysu.py and prepare\_regdb.py to prepare the training data (convert to market1501 format).

# Running
1. sh run\_train\_sysu.sh for SYSU-MM01
2. sh run\_train\_regdb.sh for RegDB
# Test
1. sh run\_test\_sysu.sh for SYSU-MM01
2. sh run\_test\_regdb.sh for RegDB


# Citation
@inproceedings{adca,
  title={Augmented Dual-Contrastive Aggregation Learning for Unsupervised Visible-Infrared Person Re-Identification},
  author={Yang, Bin and Ye, Mang and Chen, Jun and Wu, Zesen},
  pages = {2843–2851},
  booktitle = {ACM MM},
  year={2022}
}

# Contact
yangbin_cv@whu.edu.cn; yemang@whu.edu.cn.


