# R2TC-MIL

RRTMIL + **CRR** (cluster re-weighting) for bag classification.

Based on [RRT-MIL](https://github.com/DearCaat/RRT-MIL) (CVPR 2024). Before the final bag head, patches are k-means clustered; each cluster is scored by the shared classifier and used to re-weight instance attention.

## SLN 5-fold

```shell
python run_sln_5fold.py --feat-dir /path/to/pt_files --result-dir ./results
```

Defaults: ImageNet ResNet50-trunc 1024-d features, K=3 clusters, C16-R50 RRT hyper-params.

## Modules kept

- `modules/rrt.py` — RRTMIL + CRR
- `modules/kmeans.py` — CRR clustering
- `modules/rmsa.py`, `emb_position.py`, `datten.py`, `nystrom_attention.py` — RRT dependencies

## Upstream citation

```
@InProceedings{tang2024feature,
    author    = {Tang, Wenhao and Zhou, Fengtao and Huang, Sheng and Zhu, Xiang and Zhang, Yi and Liu, Bo},
    title     = {Feature Re-Embedding: Towards Foundation Model-Level Performance in Computational Pathology},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {11343-11352}
}
```
