# MNIST Neural Network from Scratch
A feedforward neural network built **entirely from scratch using only NumPy** — no PyTorch, no TensorFlow, no shortcuts. Trained on the classic MNIST handwritten digit dataset.

---

## Why I built this
I'm self-taught and wanted to actually *understand* what happens inside a neural network.
The data loading part was done with AI assistance (noted in the code), but the network architecture and training logic are mine.

---

## What it does
- Loads the MNIST dataset (~70,000 handwritten digit images)
- Trains a 3-layer neural network to classify digits 0–9
- Runs for 500 epochs with mini-batch gradient descent
- Prints training loss and accuracy each epoch
- Reports final test accuracy at the end

---

## Architecture
```
Input (784)  →  Layer 1 (128, ReLU)  →  Layer 2 (64, ReLU)  →  Output (10, Softmax)
```
- **Loss function:** Categorical Cross-Entropy
- **Optimizer:** SGD (learning rate = 0.01)
- **Batch size:** 128

---

## What I implemented manually

| Component | Description |
|---|---|
| `layer` | Linear layer (weights + biases, dot product) |
| `ReLU` | Rectified Linear Unit activation |
| `softmax` | Probability output for multi-class classification |
| `LCCE` | Loss: Categorical Cross-Entropy |
| `backPropL1` | Backprop for the output layer (softmax + CCE combined) |
| `backProp` | Backprop for hidden layers (chain rule through ReLU) |

---

## How to run

**Requirements:**
```
numpy
matplotlib
```

Install with:
```bash
pip install numpy matplotlib
```

Then just run:
```bash
python mnist_nn.py
```

The dataset downloads automatically from GitHub on first run.

---

## What I learned
- How the **chain rule** actually flows backwards through a network
- Why **softmax + cross-entropy** simplify cleanly when combined in backprop
- The difference between the output layer delta and hidden layer deltas
- Why **mini-batch SGD** is more stable than pure stochastic or full-batch gradient descent
- How small things like **weight initialization** (`* 0.1`) matter a lot for training stability

---

## References
- [Neural Networks forward pass and loss calculation](https://www.youtube.com/playlist?list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3)
- [3Blue1Brown](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) — backprop intuition and calculations

---

*Built as a self-learning project. No frameworks. Just math and NumPy.*
