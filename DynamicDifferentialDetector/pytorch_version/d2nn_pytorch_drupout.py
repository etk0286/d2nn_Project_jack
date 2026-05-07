import functools
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import math
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

# -----------------------------------------------------------------------------
# Constants and training settings
# -----------------------------------------------------------------------------
SIZE = 200
HZ = 399e9
LAMBDA = 3e8 / HZ
Z = 0.03
RECT_LENGTH = 400e-6
BUFFER_SIZE = 5000
BATCH_SIZE = 32
LEARNING_RATE = 3e-3
EPOCHS = 150
TEMPERATURE = 0.1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -----------------------------------------------------------------------------
# Helpers for FFT shifting
# -----------------------------------------------------------------------------

try:
    fftshift = torch.fft.fftshift
    ifftshift = torch.fft.ifftshift
except AttributeError:
    def fftshift(x: torch.Tensor, dim=None):
        if dim is None:
            dim = tuple(range(x.ndim))
        elif isinstance(dim, int):
            dim = (dim,)
        shift = [x.size(d) // 2 for d in dim]
        return torch.roll(x, shifts=shift, dims=dim)

    def ifftshift(x: torch.Tensor, dim=None):
        if dim is None:
            dim = tuple(range(x.ndim))
        elif isinstance(dim, int):
            dim = (dim,)
        shift = [(x.size(d) + 1) // 2 for d in dim]
        return torch.roll(x, shifts=shift, dims=dim)

# -----------------------------------------------------------------------------
# Dataset and preprocessing
# -----------------------------------------------------------------------------

def binarize_tensor(x: torch.Tensor, threshold: float) -> torch.Tensor:
    return (x > threshold).float()


def pad_tensor(x: torch.Tensor, padding: int) -> torch.Tensor:
    return F.pad(x, (padding, padding, padding, padding), mode='constant', value=0.0)


class FashionMNISTComplex(Dataset):
    def __init__(self, root: str, train: bool, size: int = SIZE, threshold: float = 0.28):
        padding = (size - int(0.8 * size)) // 2
        self.transform = transforms.Compose([
            transforms.Resize((int(0.8 * size), int(0.8 * size)), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
            transforms.Lambda(functools.partial(binarize_tensor, threshold=threshold)),
            transforms.Lambda(functools.partial(pad_tensor, padding=padding)),
        ])
        self.dataset = datasets.FashionMNIST(root=root, train=train, download=True, transform=self.transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image = image.squeeze(0)
        complex_image = torch.complex(image, torch.zeros_like(image))
        return complex_image, label

# -----------------------------------------------------------------------------
# Optical propagation layers
# -----------------------------------------------------------------------------

class DiffractionLayer(nn.Module):
    def __init__(self, size: int = SIZE, hz: float = HZ, z: float = Z, rect_length: float = RECT_LENGTH):
        super().__init__()
        self.size = size
        self.layer_size = size * rect_length
        self.dx = rect_length
        self.prop_dist = z
        self.wavelength = 3e8 / hz
        self.register_buffer('propagator', self._build_propagator())
        self.phase = nn.Parameter(torch.randn(size, size, dtype=torch.float32))

    def _build_propagator(self):
        L = self.layer_size
        dx = self.dx
        prop_dist = self.prop_dist
        wavelength = self.wavelength
        fx = np.arange(-1/(2*dx), 1/(2.*dx), 1/L, dtype=np.float32)
        FX, FY = np.meshgrid(fx, fx)
        H = np.exp(-1j * np.pi * wavelength * prop_dist * (FX**2 + FY**2)).astype(np.complex64)
        H = np.fft.fftshift(H)
        return torch.from_numpy(H)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._conv2dfft(self.propagator, x)
        phase = torch.remainder(self.phase, 2 * math.pi)
        return x * torch.exp(1j * phase)

    def _conv2dfft(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        B = fftshift(B, dim=(-2, -1))
        B = torch.fft.fft2(B)
        fftAB = A * B
        out = torch.fft.ifft2(fftAB)
        return ifftshift(out, dim=(-2, -1))


class PropagationLayer(nn.Module):
    def __init__(self, size: int = SIZE, hz: float = HZ, z: float = Z, rect_length: float = RECT_LENGTH):
        super().__init__()
        self.layer_size = size * rect_length
        self.dx = rect_length
        self.prop_dist = z
        self.wavelength = 3e8 / hz
        self.register_buffer('propagator', self._build_propagator())

    def _build_propagator(self):
        L = self.layer_size
        dx = self.dx
        prop_dist = self.prop_dist
        wavelength = self.wavelength
        fx = np.arange(-1/(2*dx), 1/(2.*dx), 1/L, dtype=np.float32)
        FX, FY = np.meshgrid(fx, fx)
        H = np.exp(-1j * np.pi * wavelength * prop_dist * (FX**2 + FY**2)).astype(np.complex64)
        H = np.fft.fftshift(H)
        return torch.from_numpy(H)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._conv2dfft(self.propagator, x)

    def _conv2dfft(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        B = fftshift(B, dim=(-2, -1))
        B = torch.fft.fft2(B)
        fftAB = A * B
        out = torch.fft.ifft2(fftAB)
        return ifftshift(out, dim=(-2, -1))

# -----------------------------------------------------------------------------
# Differential detector module
# -----------------------------------------------------------------------------

class DynamicDifferentialDetector(nn.Module):
    def __init__(self, size: int = SIZE, units: int = 10, temperature: float = TEMPERATURE, dropout_rate: float = 0.2):
        super().__init__()
        self.size = size
        self.units = units
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        intensity = x.abs() ** 2
        box_h = self.size // 10
        box_w = self.size // 10
        y_step = self.size // 5
        x_step = self.size // 4

        boxes = []
        for i in range(5):
            for j in range(4):
                y_start = (i * y_step) + (y_step - box_h) // 2
                x_start = (j * x_step) + (x_step - box_w) // 2
                patch = intensity[:, y_start:y_start + box_h, x_start:x_start + box_w]
                boxes.append(patch.mean(dim=(1, 2)))

        region_means = torch.stack(boxes, dim=1)
        pos_signals = region_means[:, :10]
        neg_signals = region_means[:, 10:]
        I_diff = (pos_signals - neg_signals) / (pos_signals + neg_signals + 1e-8)
        logits = I_diff / self.temperature
        return self.dropout(logits)

# -----------------------------------------------------------------------------
# Full PyTorch D2NN model
# -----------------------------------------------------------------------------

class D2NNModel(nn.Module):
    def __init__(self, size: int = SIZE, dropout_rate: float = 0.2):
        super().__init__()
        self.layer1 = DiffractionLayer(size=size)
        self.layer2 = DiffractionLayer(size=size)
        self.layer3 = DiffractionLayer(size=size)
        self.layer4 = DiffractionLayer(size=size)
        self.layer5 = DiffractionLayer(size=size)
        self.propagation = PropagationLayer(size=size, z=0.01)
        self.detector = DynamicDifferentialDetector(size=size, dropout_rate=dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.propagation(x)
        logits = self.detector(x)
        return logits

# -----------------------------------------------------------------------------
# Training and evaluation utilities
# -----------------------------------------------------------------------------

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        preds = torch.argmax(logits, dim=1)
        running_loss += loss.item() * images.size(0)
        running_corrects += torch.sum(preds == labels).item()
        total += images.size(0)

    return running_loss / total, running_corrects / total


def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            preds = torch.argmax(logits, dim=1)
            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels).item()
            total += images.size(0)

    return running_loss / total, running_corrects / total

# -----------------------------------------------------------------------------
# Weight extraction for phase-to-height conversion
# -----------------------------------------------------------------------------

def save_height_maps(model: D2NNModel, filename: str = 'height_map.npy'):
    phase_arrays = []
    for layer in [model.layer1, model.layer2, model.layer3, model.layer4, model.layer5]:
        phase = torch.remainder(layer.phase.detach().cpu(), 2 * math.pi).numpy()
        phase_arrays.append(phase)

    delta_n = 1.7227 - 1.0003
    height_map = (LAMBDA * np.stack(phase_arrays, axis=0)) / (2 * np.pi * delta_n)
    np.save(filename, height_map)
    print(f'Saved height map to {filename}')


def get_detector_coords(size: int):
    box_h = size // 10
    box_w = size // 10
    y_step = size // 5
    x_step = size // 4
    coords = []
    for i in range(5):
        for j in range(4):
            y_start = (i * y_step) + (y_step - box_h) // 2
            x_start = (j * x_step) + (x_step - box_w) // 2
            coords.append((x_start, y_start, box_w, box_h))
    return coords


def forward_propagation(model: D2NNModel, x: torch.Tensor) -> torch.Tensor:
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    x = model.layer5(x)
    x = model.propagation(x)
    return x


def plot_training_curves(train_losses, train_accs, val_losses, val_accs):
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot losses
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # Plot accuracies
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    filename = 'training_curves.png'
    fig.savefig(filename, dpi=150)
    print(f'Saved training curves to {filename}')
    plt.close(fig)


def plot_detector_outputs(model: D2NNModel, data_loader: DataLoader, device: torch.device, num_images: int = 6):
    model.eval()
    coords = get_detector_coords(SIZE)

    images_list = []
    labels_list = []
    for batch_images, batch_labels in data_loader:
        images_list.append(batch_images)
        labels_list.append(batch_labels)
        if sum(img.shape[0] for img in images_list) >= num_images:
            break

    if not images_list:
        raise ValueError('No images found in the data loader.')

    images = torch.cat(images_list, dim=0)[:num_images].to(device)
    labels = torch.cat(labels_list, dim=0)[:num_images].cpu().numpy()

    with torch.no_grad():
        field = forward_propagation(model, images)
        intensity = field.abs() ** 2

    intensity_np = intensity.cpu().numpy()
    images_np = images.cpu().numpy()

    fig, axes = plt.subplots(num_images, 3, figsize=(14, 4 * num_images))
    if num_images == 1:
        axes = np.expand_dims(axes, axis=0)

    for idx in range(num_images):
        input_img = np.real(images_np[idx]).astype(np.float32)
        final_intensity = intensity_np[idx]
        actual_label = int(labels[idx])

        region_means = []
        for x_start, y_start, w, h in coords:
            patch = final_intensity[y_start:y_start + h, x_start:x_start + w]
            region_means.append(patch.mean())

        region_means = np.array(region_means)
        pos_vals = region_means[:10]
        neg_vals = region_means[10:]
        diff_vals = (pos_vals - neg_vals) / (pos_vals + neg_vals + 1e-8)
        pred_label = int(np.argmax(diff_vals))
        print(f'Image {idx}: Ground truth = {actual_label}, Prediction = {pred_label}')

        ax0 = axes[idx, 0]
        ax0.imshow(input_img, cmap='gray')
        ax0.set_title(f'Input #{idx}', fontsize=12)
        ax0.text(0.02, 0.95, f'GT={actual_label}  Pred={pred_label}', transform=ax0.transAxes,
                 color='white', fontsize=10, fontweight='bold',
                 bbox=dict(facecolor='black', alpha=0.6, pad=3))
        ax0.axis('off')

        ax1 = axes[idx, 1]
        vmax = np.percentile(final_intensity, 99)
        ax1.imshow(final_intensity, cmap='viridis', vmax=vmax)
        for i, (x_start, y_start, w, h) in enumerate(coords):
            edge = 'red' if i < 10 else 'gold'
            rect = patches.Rectangle((x_start, y_start), w, h, linewidth=1.0, edgecolor=edge, facecolor='none')
            ax1.add_patch(rect)
        ax1.text(0.02, 0.95, f'GT={actual_label}', transform=ax1.transAxes,
                 color='white', fontsize=10, fontweight='bold',
                 bbox=dict(facecolor='black', alpha=0.6, pad=3))
        ax1.set_title('Final intensity + detector regions')
        ax1.axis('off')

        ax2 = axes[idx, 2]
        x_idx = np.arange(10)
        ax2.bar(x_idx + 0.2, pos_vals, width=0.4, color='red', alpha=0.6, label='pos')
        ax2.bar(x_idx + 0.2, -neg_vals, width=0.4, color='gold', alpha=0.6, label='neg')
        ax2.plot(x_idx, diff_vals, marker='o', color='black', label='diff')
        ax2.set_xticks(x_idx)
        ax2.set_title(f'GT={actual_label}  Pred={pred_label}')
        ax2.text(0.98, 0.85, f'GT={actual_label}', transform=ax2.transAxes,
                 ha='right', va='top', color='blue', fontsize=9,
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        ax2.text(0.98, 0.70, f'Pred={pred_label}', transform=ax2.transAxes,
                 ha='right', va='top', color='green', fontsize=9,
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        ax2.legend(fontsize=8)

    plt.tight_layout()
    filename = 'd2nn_detector_plot.png'
    fig.savefig(filename, dpi=150)
    print(f'Saved detector plot to {filename}')
    plt.close(fig)

# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    print('Starting D2NN PyTorch script...', flush=True)
    print(f'PyTorch version: {torch.__version__}', flush=True)
    print(f'CUDA available: {torch.cuda.is_available()}', flush=True)
    if torch.cuda.is_available():
        print(f'CUDA version: {torch.version.cuda}', flush=True)
        print(f'GPU device name: {torch.cuda.get_device_name(0)}', flush=True)
    print(f'Using device: {DEVICE}', flush=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision('high')
        print('Enabled high float32 matmul precision.', flush=True)
    except Exception:
        pass

    train_dataset = FashionMNISTComplex(root='./data', train=True)
    test_dataset = FashionMNISTComplex(root='./data', train=False)
    print(f'Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}', flush=True)

    num_workers = min(4, os.cpu_count() or 1)
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2,
    )
    print(f'Data loaders created. num_workers={num_workers}, pin_memory={pin_memory}', flush=True)

    model = D2NNModel(size=SIZE).to(DEVICE)
    print('Model created.', flush=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)

    checkpoint_dir = './training_results'
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, 'd2nn_pytorch.pt')

    # Initialize lists to record training metrics
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    start_time = time.time()
    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        print(f'--- Epoch {epoch}/{EPOCHS} ---', flush=True)
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_acc = evaluate(model, test_loader, criterion, DEVICE)
        scheduler.step()
        epoch_time = time.time() - epoch_start
        elapsed = time.time() - start_time
        remaining = epoch_time * (EPOCHS - epoch)

        # Record metrics
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f'Epoch {epoch}/{EPOCHS} '
              f'- train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} '
              f'- val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}', flush=True)
        print(f'  epoch time: {epoch_time:.1f}s, elapsed: {elapsed:.1f}s, estimated remaining: {remaining:.1f}s', flush=True)

        torch.save(model.state_dict(), checkpoint_path)

    total_time = time.time() - start_time
    save_height_maps(model, filename='height_map_pytorch.npy')
    print(f'Training complete. Total time: {total_time:.1f}s', flush=True)

    # Plot training curves
    plot_training_curves(train_losses, train_accs, val_losses, val_accs)

    plot_detector_outputs(model, test_loader, DEVICE, num_images=6)
