from pathlib import Path
import gzip
import struct
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "exemplos/26-pca-svm/data/fashion_mnist_sample.npz"


def read_images(path):
    with gzip.open(path, "rb") as stream:
        magic, n, rows, cols = struct.unpack(">IIII", stream.read(16))
        if magic != 2051:
            raise ValueError("Arquivo de imagens IDX inválido")
        return np.frombuffer(stream.read(), dtype=np.uint8).reshape(n, rows, cols)


def read_labels(path):
    with gzip.open(path, "rb") as stream:
        magic, n = struct.unpack(">II", stream.read(8))
        if magic != 2049:
            raise ValueError("Arquivo de rótulos IDX inválido")
        return np.frombuffer(stream.read(), dtype=np.uint8)


images = read_images("/tmp/fashion-images.gz")
labels = read_labels("/tmp/fashion-labels.gz")
rng = np.random.default_rng(20260827)

# Amostra balanceada: 600 imagens por classe.
indices = np.concatenate([
    rng.choice(np.where(labels == label)[0], size=600, replace=False)
    for label in range(10)
])
rng.shuffle(indices)
OUT.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(OUT, images=images[indices], labels=labels[indices])
print(OUT, OUT.stat().st_size)
