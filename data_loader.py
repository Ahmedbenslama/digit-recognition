import gzip
import os
import time
import urllib.request
import numpy as np
from config import INPUT_SIZE, NUM_CLASSES

# Fallback mirrors in case one is down or slow
MIRRORS = [
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://raw.githubusercontent.com/fgnt/mnist/master/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
]

FILES = {
    "x_train": ("train-images-idx3-ubyte.gz", 9912422),
    "y_train": ("train-labels-idx1-ubyte.gz", 28881),
    "x_test": ("t10k-images-idx3-ubyte.gz", 1648877),
    "y_test": ("t10k-labels-idx1-ubyte.gz", 4542),
}


def download_file(filename: str, expected_size: int) -> str:
    """Downloads an MNIST file with retry logic and multi-mirror support."""
    # Check if a valid, full version already exists locally
    if os.path.exists(filename):
        if os.path.getsize(filename) == expected_size:
            return filename
        else:
            print(f"Removing incomplete file {filename}...")
            os.remove(filename)

    tmp_filename = filename + ".tmp"

    for mirror in MIRRORS:
        url = mirror + filename
        print(f"Downloading {filename} from mirror: {mirror}")

        for attempt in range(1, 4):
            try:
                req = urllib.request.Request(
                    url, headers={"User-Agent": "Mozilla/5.0"}
                )
                with urllib.request.urlopen(req, timeout=20) as response, open(
                    tmp_filename, "wb"
                ) as out_file:
                    chunk_size = 64 * 1024
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        out_file.write(chunk)

                if os.path.getsize(tmp_filename) == expected_size:
                    os.rename(tmp_filename, filename)
                    print(f"Successfully downloaded {filename}")
                    return filename
                else:
                    print(f"Incomplete download (Attempt {attempt}/3). Retrying...")
            except Exception as e:
                print(f"Attempt {attempt}/3 failed: {e}")
                time.sleep(1)
            finally:
                if os.path.exists(tmp_filename):
                    os.remove(tmp_filename)

    raise RuntimeError(
        f"Failed to download {filename}. Please check your internet connection."
    )


def load_mnist_images(filename: str, expected_size: int) -> np.ndarray:
    filepath = download_file(filename, expected_size)
    with gzip.open(filepath, "rb") as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, INPUT_SIZE)


def load_mnist_labels(filename: str, expected_size: int) -> np.ndarray:
    filepath = download_file(filename, expected_size)
    with gzip.open(filepath, "rb") as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data


def one_hot_encode(labels: np.ndarray, num_classes: int = NUM_CLASSES) -> np.ndarray:
    one_hot = np.zeros((labels.size, num_classes))
    one_hot[np.arange(labels.size), labels] = 1.0
    return one_hot


def load_data():
    x_train_raw = load_mnist_images(*FILES["x_train"])
    y_train_raw = load_mnist_labels(*FILES["y_train"])
    x_test_raw = load_mnist_images(*FILES["x_test"])
    y_test_raw = load_mnist_labels(*FILES["y_test"])

    x_train = x_train_raw.astype(np.float32) / 255.0
    x_test = x_test_raw.astype(np.float32) / 255.0

    y_train_onehot = one_hot_encode(y_train_raw, NUM_CLASSES)
    y_test_onehot = one_hot_encode(y_test_raw, NUM_CLASSES)

    return (x_train, y_train_onehot), (x_test, y_test_onehot)