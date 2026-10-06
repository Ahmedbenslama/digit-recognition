import numpy as np
import matplotlib.pyplot as plt
from config import INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES, MODEL_SAVE_PATH
from data_loader import load_data
from model import NeuralNetwork, DenseLayer


def load_trained_model(weights_path: str) -> NeuralNetwork:
    """Reconstructs the model architecture and restores saved weight/bias arrays."""
    model = NeuralNetwork()
    model.add(DenseLayer(INPUT_SIZE, HIDDEN_SIZE, activation="relu"))
    model.add(DenseLayer(HIDDEN_SIZE, NUM_CLASSES, activation="softmax"))

    # Load saved .npz archive
    weights_data = np.load(weights_path)

    # Assign saved parameters back into layer objects
    for i, layer in enumerate(model.layers):
        layer.weights = weights_data[f"w_{i}"]
        layer.bias = weights_data[f"b_{i}"]

    return model


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """Computes a 10x10 confusion matrix without external library dependencies."""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        cm[true_label, pred_label] += 1
    return cm


def plot_confusion_matrix(cm: np.ndarray):
    """Renders a color-coded confusion matrix heatmap using Matplotlib."""
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, cmap="Blues")

    # Add colorbar
    plt.colorbar(im, ax=ax)

    # Label axes
    classes = [str(i) for i in range(cm.shape[0])]
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)

    ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=12, fontweight="bold")
    ax.set_title("MNIST Digit Classification - Confusion Matrix", fontsize=14, fontweight="bold")

    # Annotate numeric counts inside matrix cells
    threshold = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            # Use white text on dark cells, black text on light cells for contrast
            text_color = "white" if val > threshold else "black"
            ax.text(j, i, f"{val}", ha="center", va="center", color=text_color, fontsize=9)

    plt.tight_layout()
    plt.show()


def evaluate():
    print("Loading test dataset...")
    _, (x_test, y_test_onehot) = load_data()

    print(f"Loading trained weights from '{MODEL_SAVE_PATH}'...")
    model = load_trained_model(MODEL_SAVE_PATH)

    # Forward pass on full test set
    predictions = model.forward(x_test)
    y_pred = np.argmax(predictions, axis=1)
    y_true = np.argmax(y_test_onehot, axis=1)

    # Calculate overall accuracy
    accuracy = float(np.mean(y_pred == y_true)) * 100
    print(f"\nFinal Test Accuracy: {accuracy:.2f}%\n")

    # Generate and display confusion matrix
    cm = compute_confusion_matrix(y_true, y_pred, NUM_CLASSES)
    print("Confusion Matrix Data:")
    print(cm)

    print("\nDisplaying plot...")
    plot_confusion_matrix(cm)


if __name__ == "__main__":
    evaluate()