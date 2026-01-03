import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime
import warnings

# Suppress the old sm_120 warning (no longer needed with cu128, but safe)
warnings.filterwarnings('ignore', message='.*CUDA capability sm_120.*')

print("\n" + "="*70)
print("🏁 MNIST CNN TRAINING RACE - LOCAL RTX 5060 (PyTorch)")
print("="*70)

# Device Configuration
print("\n" + "="*70)
print("DEVICE DETECTION")
print("="*70)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    device_name = f"RTX 5060 Laptop GPU ({torch.cuda.get_device_name(0)})"
    print(f"✓ CUDA Available: True")
    print(f"✓ Device: {torch.cuda.get_device_name(0)}")
    print(f"✓ CUDA Version: {torch.version.cuda}")
    print(f"✓ Compute Capability: sm_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]}0")
    print(f"✓ Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    device_name = "CPU"
    print("⚠ No CUDA device detected - Running on CPU")
print(f"✓ PyTorch version: {torch.__version__}")
print("="*70)

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Load MNIST dataset
print("\n📊 Loading MNIST dataset...")
load_start = time.time()

from torchvision import datasets, transforms

# === FIX FOR BROKEN MNIST MIRRORS IN 2026 ===
# Original mirrors are down or inaccessible
# Use reliable Google Cloud mirror maintained by CVDFoundation
datasets.MNIST.mirrors = ["https://storage.googleapis.com/cvdf-datasets/mnist/"]
# Optional: also allow fallback to another known working mirror
# datasets.MNIST.resources = [
#     ("https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz", "train-images-idx3-ubyte.gz"),
#     ("https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz", "train-labels-idx1-ubyte.gz"),
#     ("https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz", "t10k-images-idx3-ubyte.gz"),
#     ("https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz", "t10k-labels-idx1-ubyte.gz"),
# ]

transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Convert to tensors (already done by transform, but ensure format)
x_train = train_dataset.data.unsqueeze(1).float() / 255.0  # (N,1,28,28)
y_train = train_dataset.targets
x_test = test_dataset.data.unsqueeze(1).float() / 255.0
y_test = test_dataset.targets

load_time = time.time() - load_start
print(f"✓ Training samples: {len(x_train):,}")
print(f"✓ Test samples: {len(x_test):,}")
print(f"✓ Load time: {load_time:.2f}s")

# Create data loaders
BATCH_SIZE = 128
VALIDATION_SPLIT = 0.1
val_size = int(len(x_train) * VALIDATION_SPLIT)
train_size = len(x_train) - val_size

train_dataset_split = TensorDataset(x_train[:train_size], y_train[:train_size])
val_dataset_split = TensorDataset(x_train[train_size:], y_train[train_size:])
test_dataset_split = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_dataset_split, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset_split, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset_split, batch_size=BATCH_SIZE)

# Define CNN Model
print("\n🏗️ Building CNN model...")
model_start = time.time()

class MNISTCNN(nn.Module):
    def __init__(self):
        super(MNISTCNN, self).__init__()
       
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.25)
       
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.dropout2 = nn.Dropout(0.25)
       
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.dropout3 = nn.Dropout(0.25)
       
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 7 * 7, 128)
        self.dropout4 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)
       
        self.relu = nn.ReLU()
       
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.dropout1(x)
       
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.dropout2(x)
       
        x = self.relu(self.conv3(x))
        x = self.dropout3(x)
       
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout4(x)
        x = self.fc2(x)
       
        return x

model = MNISTCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

model_time = time.time() - model_start
total_params = sum(p.numel() for p in model.parameters())
print(f"✓ Model built in {model_time:.2f}s")
print(f"✓ Total parameters: {total_params:,}")

# Training configuration
EPOCHS = 10
print("\n" + "="*70)
print("🚀 STARTING TRAINING RACE")
print("="*70)
print(f"Device: {device_name}")
print(f"Epochs: {EPOCHS}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Validation Split: {VALIDATION_SPLIT}")
print("="*70)

# Training function
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
   
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
       
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
       
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
   
    return total_loss / len(loader), correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
   
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
           
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
   
    return total_loss / len(loader), correct / total

# Record start time
race_start = time.time()
training_start = datetime.now()
print(f"\n⏱️ Race Start Time: {training_start.strftime('%H:%M:%S.%f')[:-3]}")
print("\n" + "-"*70)

# Training loop
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
for epoch in range(EPOCHS):
    epoch_start = time.time()
   
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
   
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
   
    epoch_time = time.time() - epoch_start
   
    print(f"Epoch {epoch+1}/{EPOCHS} - {epoch_time:.2f}s - "
          f"loss: {train_loss:.4f} - accuracy: {train_acc:.4f} - "
          f"val_loss: {val_loss:.4f} - val_accuracy: {val_acc:.4f}")

# Record end time
training_end = datetime.now()
race_end = time.time()
total_training_time = race_end - race_start
print("\n" + "-"*70)
print(f"⏱️ Race End Time: {training_end.strftime('%H:%M:%S.%f')[:-3]}")
print("="*70)

# Evaluate on test set
print("\n📈 EVALUATING MODEL")
print("="*70)
eval_start = time.time()
test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
eval_time = time.time() - eval_start

# Make sample predictions
num_samples = 10
sample_indices = np.random.choice(len(x_test), num_samples, replace=False)
sample_images = x_test[sample_indices].to(device)
sample_labels = y_test[sample_indices].numpy()

pred_start = time.time()
model.eval()
with torch.no_grad():
    predictions = model(sample_images)
    predicted_labels = predictions.argmax(dim=1).cpu().numpy()
pred_time = time.time() - pred_start

correct_count = sum(predicted_labels == sample_labels)

# Final Results
print("\n" + "="*70)
print("🏆 FINAL RACE RESULTS")
print("="*70)
print(f"Device: {device_name}")
print(f"PyTorch: {torch.__version__}")
print("-"*70)
print(f"Total Training Time: {total_training_time:.2f} seconds")
print(f" ({total_training_time/60:.2f} minutes)")
print(f"Average per Epoch: {total_training_time/EPOCHS:.2f} seconds")
print("-"*70)
print(f"Test Accuracy: {test_accuracy*100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")
print(f"Evaluation Time: {eval_time:.2f}s")
print(f"Inference Time: {pred_time*1000:.2f}ms ({num_samples} images)")
print(f"Per Image: {(pred_time/num_samples)*1000:.2f}ms")
print(f"Sample Accuracy: {correct_count}/{num_samples}")
print("-"*70)
print(f"Race Started: {training_start.strftime('%H:%M:%S')}")
print(f"Race Finished: {training_end.strftime('%H:%M:%S')}")
print(f"Duration: {training_end - training_start}")
print("="*70)

# Visualize sample predictions
print("\n📊 Generating visualizations...")
plt.figure(figsize=(15, 6))
for i in range(num_samples):
    plt.subplot(2, 5, i+1)
    plt.imshow(sample_images[i].cpu().squeeze(), cmap='gray')
    color = 'green' if predicted_labels[i] == sample_labels[i] else 'red'
    plt.title(f"Pred: {predicted_labels[i]}\nTrue: {sample_labels[i]}",
              color=color, fontsize=9)
    plt.axis('off')
plt.suptitle(f'Sample Predictions - {device_name}', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('predictions_race_pytorch.png', dpi=150, bbox_inches='tight')
print("✓ Saved: predictions_race_pytorch.png")

# Plot training history
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
epochs_range = range(1, EPOCHS + 1)
ax1.plot(epochs_range, history['train_acc'], label='Training', linewidth=2, marker='o')
ax1.plot(epochs_range, history['val_acc'], label='Validation', linewidth=2, marker='s')
ax1.set_title(f'Accuracy - {device_name}', fontsize=14, fontweight='bold')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

ax2.plot(epochs_range, history['train_loss'], label='Training', linewidth=2, marker='o')
ax2.plot(epochs_range, history['val_loss'], label='Validation', linewidth=2, marker='s')
ax2.set_title(f'Loss - {device_name}', fontsize=14, fontweight='bold')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Loss', fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.suptitle(f'Training History - Total Time: {total_training_time:.2f}s',
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('training_history_pytorch.png', dpi=150, bbox_inches='tight')
print("✓ Saved: training_history_pytorch.png")

# Save model
torch.save(model.state_dict(), 'mnist_race_pytorch.pth')
print("✓ Model saved: mnist_race_pytorch.pth")

print("\n" + "="*70)
print("✅ RACE COMPLETE!")
print("="*70)
print(f"⏱️ Your time: {total_training_time:.2f} seconds")
print(f"🎯 Accuracy: {test_accuracy*100:.2f}%")
print("\nBlackwell power unleashed! 🚀")
print("="*70)
