import time
import numpy as np
from config import (
    INPUT_SIZE,
    HIDDEN_SIZE,
    NUM_CLASSES,
    LEARNING_RATE,
    BATCH_SIZE,
    EPOCHS,
    MODEL_SAVE_PATH,
)
from data_loader import load_data
from model import NeuralNetwork, DenseLayer


def compute_accuracy(predictions: np.ndarray, targets_onehot: np.ndarray) -> float:
    """Calculates accuracy by comparing argmax predictions against true labels."""
    pred_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(targets_onehot, axis=1)
    return float(np.mean(pred_labels == true_labels))


def train():
    # 1. Load Preprocessed Data
    print("Loading dataset...")
    (x_train, y_train), (x_test, y_test) = load_data()
    num_samples = x_train.shape[0]

    # 2. Build Model Architecture
    model = NeuralNetwork()
    model.add(DenseLayer(INPUT_SIZE, HIDDEN_SIZE, activation="relu"))
    model.add(DenseLayer(HIDDEN_SIZE, NUM_CLASSES, activation="softmax"))

    print(
        f"Starting training ({EPOCHS} epochs, batch_size={BATCH_SIZE}, lr={LEARNING_RATE})...\n"
    )

    # 3. Epoch Loop
    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()

        # Shuffle indices at the start of every epoch to prevent mini-batch bias
        indices = np.random.permutation(num_samples)
        x_train_shuffled = x_train[indices]
        y_train_shuffled = y_train[indices]

        total_epoch_loss = 0.0
        num_batches = int(np.ceil(num_samples / BATCH_SIZE))

        # Mini-Batch SGD Loop
        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min(start_idx + BATCH_SIZE, num_samples)

            x_batch = x_train_shuffled[start_idx:end_idx]
            y_batch = y_train_shuffled[start_idx:end_idx]

            # Forward pass, loss calculation, backpropagation, and weight update
            batch_loss = model.train_step(x_batch, y_batch, LEARNING_RATE)
            total_epoch_loss += batch_loss * (end_idx - start_idx)

        # Average loss across all training samples
        avg_loss = total_epoch_loss / num_samples

        # Calculate epoch accuracies
        train_preds = model.forward(x_train)
        train_acc = compute_accuracy(train_preds, y_train)

        test_preds = model.forward(x_test)
        test_acc = compute_accuracy(test_preds, y_test)

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch:02d}/{EPOCHS:02d} | "
            f"Loss: {avg_loss:.4f} | "
            f"Train Acc: {train_acc * 100:.2f}% | "
            f"Test Acc: {test_acc * 100:.2f}% | "
            f"Time: {elapsed:.2f}s"
        )

    # 4. Save Trained Weights and Biases
    save_dict = {}
    for i, layer in enumerate(model.layers):
        save_dict[f"w_{i}"] = layer.weights
        save_dict[f"b_{i}"] = layer.bias

    # Save to compressed NumPy archive file (.npz)
    np.savez(MODEL_SAVE_PATH, **save_dict)
    print(f"\nTrained model weights successfully saved to '{MODEL_SAVE_PATH}'!")


if __name__ == "__main__":
    train()