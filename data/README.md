# REAL DI CHAT dataset

Put UTF-8 `.txt` files containing the training corpus in this directory.

Recommended layout:

```text
data/
  train.txt
  val.txt
```

`train.txt` is used for optimization and `val.txt` is used only for validation. If `val.txt` is absent, `train.py` creates a deterministic 90/10 split from `train.txt`.

For useful language-model training, use a substantially larger and legally usable corpus than the tiny demo text shipped with the repository. Plain text is intentionally supported first so the data pipeline stays inspectable.
