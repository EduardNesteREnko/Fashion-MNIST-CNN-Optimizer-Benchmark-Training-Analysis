# Fashion-MNIST CNN Optimizer Benchmark & Training Analysis

This repository contains a PyTorch implementation of a Convolutional Neural Network (CNN) trained on the **Fashion-MNIST** dataset. The script benchmarks and compares the convergence and performance of four different optimization algorithms (`Adam`, `RMSprop`, `SGD with Nesterov Momentum`, and `SGD with Momentum`).

## 📋 Pipeline Overview

1. **Dataset & Preprocessing**:
   * Loads the `FashionMNIST` dataset via `torchvision.datasets`.
   * Applies tensor transformation and normalization (`mean=0.5, std=0.5`) across images.
   * Configures `DataLoader` instances with a batch size of 128 for both training and testing splits.

2. **Convolutional Neural Network Architecture (`FashionCNN`)**:
   * **Feature Extractor**: Two sequential blocks consisting of 2D convolutions ($3\times3$ kernel, padding=1), ReLU activation functions, and Max Pooling ($2\times2$, stride=2).
   * **Classifier**: Fully connected layers with dropout regularization ($p=0.3$) and linear mappings down to 10 output classes.

3. **Optimizer Comparative Benchmark**:
   * Automatically initializes separate model instances for each optimizer to ensure isolated training paths.
   * Trains each model for 15 epochs using `CrossEntropyLoss`.
   * Evaluates:
     * **Adam** ($\text{lr}=0.001$)
     * **RMSprop** ($\text{lr}=0.001$)
     * **SGD + Nesterov Momentum** ($\text{lr}=0.01, \text{momentum}=0.9$)
     * **SGD + Momentum** ($\text{lr}=0.01, \text{momentum}=0.9$)

4. **Visualization**:
   * Plots a logarithmic training cost (loss) curve across epochs for all optimizers and exports the final graphic as `fashion_mnist_pytorch_optimizers.png`.

---

## 🛠️ Code Implementation

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=128, shuffle=False)

# Convolutional Neural Network Architecture
class FashionCNN(nn.Module):
    def __init__(self):
        super(FashionCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def get_optimizers(model):
    return {
        'Adam': optim.Adam(model.parameters(), lr=0.001),
        'RMSprop': optim.RMSprop(model.parameters(), lr=0.001),
        'SGDNesterov': optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True),
        'SGM (SGD + Momentum)': optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    }

epochs = 15
criterion = nn.CrossEntropyLoss()
history_results = {}

temp_model = FashionCNN()
optimizers_dict = get_optimizers(temp_model)

for opt_name, _ in optimizers_dict.items():
    model = FashionCNN().to(device)

    if opt_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    elif opt_name == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=0.001)
    elif opt_name == 'SGDNesterov':
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
    elif opt_name == 'SGM (SGD + Momentum)':
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    epoch_losses = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    history_results[opt_name] = epoch_losses

# Logarithmic training loss curve
plt.figure(figsize=(10, 7))

for opt_name, losses in history_results.items():
    plt.plot(losses, label=opt_name, linewidth=2)

plt.yscale('log')
plt.title('Fashion-MNIST CNN: Training Cost (Loss) by Optimizer', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Training Cost (Loss)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=11)
plt.tight_layout()

plt.savefig('fashion_mnist_pytorch_optimizers.png', dpi=300)
plt.show()
```

---

## 📊 Results & Conclusions

* **Top Performers**: **RMSprop** ($Loss \approx 0.1010$) and **Adam** ($Loss \approx 0.1045$) delivered the lowest training losses by the 15th epoch, demonstrating superior adaptive gradient convergence speed.
* **Baseline Momentum**: **SGM** ($Loss \approx 0.1780$) and **SGDNesterov** ($Loss \approx 0.1790$) performed moderately worse, showing slower descent trajectories under the specified learning rate schedule.