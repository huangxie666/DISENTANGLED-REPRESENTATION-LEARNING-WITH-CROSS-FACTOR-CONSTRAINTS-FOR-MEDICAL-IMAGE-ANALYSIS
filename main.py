# main.py
import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn.functional as F
from data_loader import get_data_loaders
from model import CompleteModel
import numpy as np
from train import train_model
from visualize import plot_training_results, plot_confusion_matrix, visualize_predictions
import os
import json
import sys
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

def set_seed(seed=42):
    np.random.seed(seed)  # Numpy 随机种子
    torch.manual_seed(seed)  # CPU 上的 PyTorch 随机种子
    torch.cuda.manual_seed(seed)  # GPU 上的 PyTorch 随机种子
    torch.cuda.manual_seed_all(seed)  # 多 GPU 设置

    # MPS/CPU 确保完全可复现（但速度可能下降）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
class DualLogger:
    def __init__(self, file_path):
        self.terminal = sys.__stdout__
        self.log_file = open(file_path, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

class LossCalculator:
    def __init__(self, temperature=0.1):
        self.temperature = temperature

    def contrastive_loss(self, proj_feat1, proj_feat2):
        batch_size = proj_feat1.size(0)
        z1 = F.normalize(proj_feat1, dim=1)
        z2 = F.normalize(proj_feat2, dim=1)
        logits = torch.mm(z1, z2.T) / self.temperature
        labels = torch.arange(batch_size, device=proj_feat1.device)
        return F.cross_entropy(logits, labels)

    def correlation_loss(self, feat1, feat3):
        feat1 = feat1 - feat1.mean(dim=0, keepdim=True)
        feat3 = feat3 - feat3.mean(dim=0, keepdim=True)
        cov = (feat1 * feat3).mean(dim=0)
        var1 = feat1.pow(2).mean(dim=0)
        var3 = feat3.pow(2).mean(dim=0)
        eps = 1e-8
        denominator = torch.sqrt(var1) * torch.sqrt(var3) + eps
        return (torch.abs(cov) / denominator).mean()

    def reconstruction_loss(self, recon, target):
        return F.mse_loss(recon, target)

    def feature_contrastive_loss(self, x1, x2):
        return self.contrastive_loss(x1.view(x1.size(0), -1), x2.view(x2.size(0), -1))

def setup_device():
    if torch.cuda.is_available():
        print("Using GPU (CUDA)")
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("Using Apple Silicon (MPS)")
        return torch.device("mps")
    else:
        print("Using CPU")
        return torch.device("cpu")

def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)
    print(f"Checkpoint saved to {path}")

def main():
    parser = argparse.ArgumentParser(description='Breast Ultrasound Classification')
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--output_dir', default='./results', type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--lambda_corr', default=0.001, type=float)
    parser.add_argument('--lambda_ortho', default=0.0001, type=float)
    parser.add_argument('--lambda_match', default=0.001, type=float)
    parser.add_argument('--lambda_recon', default=0.00001, type=float)
    parser.add_argument('--lambda_contrast', default=0.1, type=float)
    parser.add_argument('--resume', default=None, type=str)
    parser.add_argument('--no_classification', default=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, 'log.txt')
    sys.stdout = sys.stderr = DualLogger(log_path)

    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    device = setup_device()
    root_dir = "/Users/huangxiaoxie/Desktop/data/Breast_Ultrasound_Images/Dataset_BUSI_with_GT"
    train_loader, test_loader, classes = get_data_loaders(root_dir, args.batch_size)

    if args.no_classification:
        model = CompleteModel(num_classes=None).to(device)
    else:
        model = CompleteModel(num_classes=len(classes)).to(device)

    criterion = nn.CrossEntropyLoss()
    loss_calculator = LossCalculator()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch}")

    results = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=args.num_epochs,
        loss_calculator=loss_calculator,
        lambda_corr=args.lambda_corr,
        lambda_ortho=args.lambda_ortho,
        lambda_match=args.lambda_match,
        lambda_recon=args.lambda_recon,
        lambda_contrast=args.lambda_contrast,
        start_epoch=start_epoch
    )

    final_model_path = os.path.join(args.output_dir, 'final_model.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")

    plot_training_results(
        results['train_loss'], 
        results['test_acc'],
        save_path=os.path.join(args.output_dir, 'training_plot.png')
    )
    plot_confusion_matrix(
        results['all_labels'],
        results['all_preds'],
        classes,
        save_path=os.path.join(args.output_dir, 'confusion_matrix.png')
    )
    visualize_predictions(
        model=model,
        test_loader=test_loader,
        class_names=classes,
        device=device,
        save_dir=args.output_dir,
        num_examples=5
    )

if __name__ == '__main__':
    main()

def plot_training_results(train_loss, test_acc, save_path=None):
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='Train Loss')
    plt.plot(test_acc, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
        print(f"Training curve saved to: {save_path}")
    else:
        plt.show()