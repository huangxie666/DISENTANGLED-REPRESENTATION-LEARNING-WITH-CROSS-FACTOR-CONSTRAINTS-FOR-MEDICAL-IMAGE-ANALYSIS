# visualize.py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch
import os
from torchvision.utils import make_grid
import torchvision.transforms.functional as TF

def plot_training_results(train_loss_history, test_acc_history, save_path=None):
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_history, label='Training Loss')
    plt.plot(test_acc_history, label='Test Accuracy', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.title('Training Loss & Accuracy')
    plt.legend()
    plt.grid()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved training curve to {save_path}")
    else:
        plt.show()


def plot_confusion_matrix(all_labels, all_preds, classes, save_path=None):
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    plt.title("Confusion Matrix")
    if save_path:
        plt.savefig(save_path)
        print(f"Saved confusion matrix to {save_path}")
    else:
        plt.show()


def visualize_predictions(model, dataset, classes, device, save_dir="./", num_images=6):
    model.eval()
    indices = np.random.choice(len(dataset), num_images, replace=False)
    os.makedirs(save_dir, exist_ok=True)

    for i, idx in enumerate(indices):
        (x1, x2, x3), label = dataset[idx]
        x1 = x1.unsqueeze(0).to(device)
        x2 = x2.unsqueeze(0).to(device)
        x3 = x3.unsqueeze(0).to(device)

        with torch.no_grad():
            output_dict = model(x1, x2, x3)
            logits = output_dict.get("logits", None)
            recon = output_dict.get("reconstruction", None)
            if logits is not None:
                _, predicted = torch.max(logits.data, 1)
            else:
                predicted = torch.tensor([0])  # dummy

        # 原图
        image = x1.squeeze().cpu().numpy()
        image = np.transpose(image, (1, 2, 0))
        image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        image = np.clip(image, 0, 1)

        plt.figure(figsize=(6, 3))
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title(f"True: {classes[label]}\nPred: {classes[predicted.item()]}")
        plt.axis('off')

        if recon is not None:
            recon_img = recon.squeeze().cpu().numpy()
            recon_img = np.transpose(recon_img, (1, 2, 0))
            recon_img = np.clip(recon_img, 0, 1)

            plt.subplot(1, 2, 2)
            plt.imshow(recon_img)
            plt.title("Reconstruction")
            plt.axis('off')

        plt.tight_layout()
        save_path = os.path.join(save_dir, f"prediction_{i+1}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved prediction example to {save_path}")