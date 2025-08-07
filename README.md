# 🧠 Spiking Neural Networks (SNN) for Classification using snnTorch

This project demonstrates the implementation and training of a **Spiking Neural Network (SNN)** on the **MNIST** dataset using the `snntorch` library. Inspired by the behavior of biological neurons, SNNs are energy-efficient and time-aware neural networks, ideal for sequential data processing.

---

## Overview

- **ANN (Artificial Neural Network)**: Uses continuous (numeric) signals.
- **SNN (Spiking Neural Network)**: Uses discrete spikes that mimic real neuron firing.
- **Neuron Model Used**: Leaky Integrate-and-Fire (LIF)
- **Dataset**: [MNIST Handwritten Digits](http://yann.lecun.com/exdb/mnist/)
- **Library**: [`snntorch`](https://snntorch.readthedocs.io/)

---

## Project Structure

1. **Data Loading**  
   Loads the MNIST dataset using `torchvision.datasets` and converts it to tensors.

2. **Model Definition**  
   A fully connected 2-layer SNN with LIF neurons:
   ```python
   fc1 = nn.Linear(784, 512)
   lif1 = snn.Leaky(beta=0.5)
   fc2 = nn.Linear(512, 10)
   lif2 = snn.Leaky(beta=0.5)
   ```

3. **Training**
   - Inputs are propagated through the network for a fixed number of time steps.
   - Spike outputs are averaged over time.
   - Uses `CrossEntropyLoss` and `Adam` optimizer.
   - Surrogate gradients are applied to enable training.

4. **Evaluation**
   - Final predictions are obtained by averaging spikes across time.
   - Model accuracy is calculated on the test set.

5. **Confusion Matrix**
   - A normalized confusion matrix is visualized using `seaborn`.

## 📈 Results

- **Accuracy**: ~96-98% on MNIST after a few epochs
- **Confusion Matrix**: Visualizes performance across digit classes

