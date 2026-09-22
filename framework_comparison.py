"""
PyTorch vs. TensorFlow: The Same Task, Two Frameworks
========================================================
This project trains the same small classifier on the same synthetic
data using TensorFlow/Keras, actually run in this environment with
real reported numbers. The equivalent PyTorch code is included for
direct comparison, written to current PyTorch API conventions, but
NOT executed here (disclosed honestly below), since installing full
PyTorch alongside TensorFlow exceeded this environment's available
disk space. The TensorFlow numbers reported are entirely real.

Task: classify 2D points into 2 classes (a simple non-linear
boundary), the same kind of toy problem used throughout this series'
from-scratch NumPy projects, so the code style differences between
frameworks are easy to compare directly against those.

Author: Khalid Hussain, ReviewPublically.com
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import time

np.random.seed(0)
tf.random.set_seed(0)

# ---------------------------------------------------------------
# 1. Same synthetic data used elsewhere in this series: a curved
#    decision boundary, some noise
# ---------------------------------------------------------------
def make_data(n, noise, seed):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.5, 1.5, (n, 2))
    y = (X[:, 1] > 0.6 * np.sin(3 * X[:, 0]) + 0.1).astype(int)
    flip = rng.uniform(size=n) < noise
    y = np.where(flip, 1 - y, y)
    return X.astype("float32"), y.astype("int32")

X_train, y_train = make_data(500, noise=0.05, seed=1)
X_test, y_test = make_data(200, noise=0.0, seed=2)

# ---------------------------------------------------------------
# 2. Build and train the model in TensorFlow / Keras. This is the
#    part of the project that is actually executed, with real
#    numbers reported below.
# ---------------------------------------------------------------
model = keras.Sequential([
    keras.layers.Input(shape=(2,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(2, activation="softmax"),
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

start = time.time()
history = model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
train_time = time.time() - start

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

print("--- TensorFlow / Keras: actually run in this environment ---")
print(f"TensorFlow version: {tf.__version__}")
print(f"Training time for 50 epochs: {train_time:.2f} seconds")
print(f"Final training loss: {history.history['loss'][-1]:.4f}")
print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_acc:.4f}")
print(f"Total trainable parameters: {model.count_params()}")

print("""
--- Equivalent PyTorch code (same architecture, same task) ---
NOT executed in this environment: installing full PyTorch alongside
TensorFlow exceeded the available disk space here (both frameworks
together need more room than this sandbox had free). The code below
is accurate to current PyTorch API conventions but its numbers are
not reported, only the TensorFlow run above produced real numbers.

    import torch
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, 16), nn.ReLU(),
                nn.Linear(16, 16), nn.ReLU(),
                nn.Linear(16, 2),
            )
        def forward(self, x):
            return self.net(x)

    model = Net()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train, dtype=torch.long)

    for epoch in range(50):
        optimizer.zero_grad()
        logits = model(X_train_t)
        loss = loss_fn(logits, y_train_t)
        loss.backward()
        optimizer.step()
""")

print("--- What the code comparison actually shows ---")
print("Keras: model is built as a sequence of declared layers, then")
print("       .fit() runs the entire training loop internally.")
print("PyTorch: the training loop (zero_grad, forward, loss, backward,")
print("       step) is written out explicitly by the developer.")
print("Neither is 'more correct'. Keras trades control for brevity.")
print("PyTorch trades brevity for direct visibility into every step,")
print("the same visibility this entire from-scratch NumPy series has")
print("been building toward by hand.")

# ---------------------------------------------------------------
# Plot: real training curve from the actual TensorFlow run
# ---------------------------------------------------------------
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(history.history['loss'], color="#0E6E56", linewidth=2)
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Training loss")
axes[0].set_title("Real TensorFlow Training Loss (50 epochs)")

axes[1].plot(history.history['accuracy'], color="#3648B8", linewidth=2)
axes[1].axhline(test_acc, color="gray", linestyle="--", label=f"Test accuracy ({test_acc:.3f})")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Training accuracy")
axes[1].set_title("Real TensorFlow Training Accuracy")
axes[1].legend()

fig.suptitle(f"Same Task Trained for Real in TensorFlow {tf.__version__} ({train_time:.1f}s, 50 epochs)")
fig.tight_layout()
fig.savefig("/home/claude/pytorch-vs-tensorflow/tensorflow_training_curve.png", dpi=150)
plt.close()

print("\nSaved: tensorflow_training_curve.png")
