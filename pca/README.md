# PCA

This folder contains the linear dimensionality reduction baseline for the
project.

- `pca_mnist.py`: runs PCA on MNIST, reconstructs held-out images, and writes
  metrics plus plots.
- `results/`: saved PCA metrics, explained-variance plots, and reconstruction
  grids.

Run it with:

```bash
python pca/pca_mnist.py
```
