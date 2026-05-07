import argparse
import os

# Workaround duplicate OpenMP runtime conflict on Windows
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
from torch.utils.data import DataLoader
from d2nn_pytorch import D2NNModel, FashionMNISTComplex, plot_detector_outputs, SIZE, DEVICE


def load_checkpoint(model: D2NNModel, checkpoint_path: str, device: torch.device):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')

    state_dict = torch.load(checkpoint_path, map_location=device)
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return state_dict


def find_misclassified_samples(model, test_loader, device, max_samples=50):
    """Find and return indices of misclassified samples."""
    model.eval()
    misclassified_indices = []
    misclassified_labels = []
    misclassified_preds = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)

            # Find misclassified samples in this batch
            incorrect_mask = preds != labels
            if incorrect_mask.any():
                batch_size = images.size(0)
                batch_start_idx = batch_idx * batch_size

                incorrect_indices = torch.where(incorrect_mask)[0]
                for idx in incorrect_indices:
                    global_idx = batch_start_idx + idx.item()
                    misclassified_indices.append(global_idx)
                    misclassified_labels.append(labels[idx].item())
                    misclassified_preds.append(preds[idx].item())

                    if len(misclassified_indices) >= max_samples:
                        return misclassified_indices, misclassified_labels, misclassified_preds

    return misclassified_indices, misclassified_labels, misclassified_preds


def plot_misclassified_samples(model, test_dataset, misclassified_indices, misclassified_labels, misclassified_preds, device, num_images=10):
    """Plot a subset of misclassified samples."""
    from d2nn_pytorch import get_detector_coords, forward_propagation
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import numpy as np

    model.eval()
    coords = get_detector_coords(200)  # SIZE = 200

    # Select subset of misclassified samples
    num_to_plot = min(num_images, len(misclassified_indices))
    selected_indices = misclassified_indices[:num_to_plot]
    selected_labels = misclassified_labels[:num_to_plot]
    selected_preds = misclassified_preds[:num_to_plot]

    fig, axes = plt.subplots(num_to_plot, 3, figsize=(14, 4 * num_to_plot))
    if num_to_plot == 1:
        axes = np.expand_dims(axes, axis=0)

    for idx, (sample_idx, true_label, pred_label) in enumerate(zip(selected_indices, selected_labels, selected_preds)):
        # Get the sample
        image, _ = test_dataset[sample_idx]
        image = image.unsqueeze(0).to(device)  # Add batch dimension

        with torch.no_grad():
            field = forward_propagation(model, image)
            intensity = field.abs() ** 2

        intensity_np = intensity.cpu().numpy()[0]
        image_np = image.cpu().numpy()[0]

        # Plot input image
        ax0 = axes[idx, 0]
        ax0.imshow(np.real(image_np).astype(np.float32), cmap='gray')
        ax0.set_title(f'Sample {sample_idx}', fontsize=12)
        ax0.text(0.02, 0.95, f'GT={true_label}  Pred={pred_label}', transform=ax0.transAxes,
                 color='red', fontsize=10, fontweight='bold',
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='red'))
        ax0.axis('off')

        # Plot final intensity with detector regions
        ax1 = axes[idx, 1]
        vmax = np.percentile(intensity_np, 99)
        ax1.imshow(intensity_np, cmap='viridis', vmax=vmax)
        for i, (x_start, y_start, w, h) in enumerate(coords):
            edge = 'red' if i < 10 else 'gold'
            rect = patches.Rectangle((x_start, y_start), w, h, linewidth=1.0, edgecolor=edge, facecolor='none')
            ax1.add_patch(rect)
        ax1.set_title('Final intensity + detector regions')
        ax1.axis('off')

        # Plot detector signals
        ax2 = axes[idx, 2]
        region_means = []
        for x_start, y_start, w, h in coords:
            patch = intensity_np[y_start:y_start + h, x_start:x_start + w]
            region_means.append(patch.mean())

        region_means = np.array(region_means)
        pos_vals = region_means[:10]
        neg_vals = region_means[10:]
        diff_vals = (pos_vals - neg_vals) / (pos_vals + neg_vals + 1e-8)

        x_idx = np.arange(10)
        ax2.bar(x_idx + 0.2, pos_vals, width=0.4, color='red', alpha=0.6, label='pos')
        ax2.bar(x_idx + 0.2, -neg_vals, width=0.4, color='gold', alpha=0.6, label='neg')
        ax2.plot(x_idx, diff_vals, marker='o', color='black', label='diff')
        ax2.set_xticks(x_idx)
        ax2.set_title(f'GT={true_label}  Pred={pred_label}')
        ax2.legend(fontsize=8)

    plt.tight_layout()
    filename = 'misclassified_samples.png'
    fig.savefig(filename, dpi=150)
    print(f'Saved misclassified samples plot to {filename}')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Load a D2NN checkpoint and plot detector output.')
    parser.add_argument('--checkpoint', default='./training_results/d2nn_pytorch.pt', help='Path to the checkpoint file')
    parser.add_argument('--data-root', default='./data', help='Root folder for FashionMNIST data')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for the test loader')
    parser.add_argument('--num-images', type=int, default=6, help='Number of images to plot')
    args = parser.parse_args()

    print(f'Using checkpoint: {args.checkpoint}')
    print(f'Using device: {DEVICE}')

    test_dataset = FashionMNISTComplex(root=args.data_root, train=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False)

    model = D2NNModel(size=SIZE).to(DEVICE)
    state_dict = load_checkpoint(model, args.checkpoint, DEVICE)

    print('Checkpoint loaded successfully.')
    print(f'Checkpoint contains {len(state_dict)} tensors')
    for key in list(state_dict.keys())[:20]:
        print(f'  {key}: {tuple(state_dict[key].shape)}')

    # Calculate overall statistics
    print('\n=== Dataset Statistics ===')
    total_samples = len(test_dataset)
    print(f'總共測試照片數量: {total_samples}')

    print('計算預測統計...')
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            correct += torch.sum(preds == labels).item()
            total += labels.size(0)

    accuracy = correct / total
    misclassified_count = total - correct
    print(f'正確預測數量: {correct}')
    print(f'辨識錯誤數量: {misclassified_count}')
    print(f'整體準確率: {accuracy:.4f} ({correct}/{total})')

    # Find misclassified samples
    print('\nFinding misclassified samples...')
    misclassified_indices, misclassified_labels, misclassified_preds = find_misclassified_samples(
        model, test_loader, DEVICE, max_samples=100
    )

    print(f'Found {len(misclassified_indices)} misclassified samples out of {len(test_dataset)} total samples')

    if misclassified_indices:
        print('Misclassified samples (first 20):')
        for i, (idx, true, pred) in enumerate(zip(misclassified_indices[:20], misclassified_labels[:20], misclassified_preds[:20])):
            print(f'  Sample {idx}: GT={true}, Pred={pred}')

        # Plot some misclassified samples
        plot_misclassified_samples(model, test_dataset, misclassified_indices, misclassified_labels,
                                 misclassified_preds, DEVICE, num_images=min(10, len(misclassified_indices)))
    else:
        print('No misclassified samples found!')

    # Also plot some general samples as before
    plot_detector_outputs(model, test_loader, DEVICE, num_images=args.num_images)


if __name__ == '__main__':
    main()
