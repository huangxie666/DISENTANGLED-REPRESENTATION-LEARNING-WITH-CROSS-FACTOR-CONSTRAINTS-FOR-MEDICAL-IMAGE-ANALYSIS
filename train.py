import torch
import numpy as np

def train_model(model, train_loader, test_loader, criterion, optimizer, device, 
                num_epochs=10, start_epoch=0, loss_calculator=None, 
                lambda_corr=0.1, lambda_ortho=0.1, lambda_match=0.1,
                lambda_recon=0.1, lambda_contrast=0.1):

    train_loss_history = []
    test_acc_history = []
    all_preds = []
    all_labels = []

    for epoch in range(start_epoch, start_epoch + num_epochs):
        model.train()
        epoch_loss = 0.0
        batch_count = 0

        for batch_idx, (x, labels) in enumerate(train_loader):
            x = x.to(device)  # 单输入
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(x)

            private_feat = outputs['Private_MLP']
            disease_feat = outputs['Disease_MLP']
            common_feat = outputs['Common_MLP']
            recon = outputs['reconstruction']
            logits = outputs.get('logits', None)

            total_loss = torch.tensor(0.0, device=device)

            # ✅ 主分类损失
            if logits is not None:
                main_loss = criterion(logits, labels)
                total_loss += main_loss

            # ✅ correlation loss
            if loss_calculator is not None:
                corr_loss = loss_calculator.correlation_loss(private_feat, common_feat)
                total_loss += lambda_corr * corr_loss

                recon_loss = loss_calculator.reconstruction_loss(recon, x)
                total_loss += lambda_recon * recon_loss

                contrast_loss = loss_calculator.feature_contrastive_loss(private_feat, disease_feat)
                total_loss += lambda_contrast * contrast_loss

            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            batch_count += 1

            if batch_count % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {epoch_loss / batch_count:.4f}")

        avg_epoch_loss = epoch_loss / batch_count
        train_loss_history.append(avg_epoch_loss)

        # ==== 测试阶段 ====
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for x, labels in test_loader:
                x = x.to(device)
                labels = labels.to(device)

                outputs = model(x)
                logits = outputs.get('logits', None)

                if logits is not None:
                    _, predicted = torch.max(logits, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

        if total > 0:
            test_acc = 100 * correct / total
            test_acc_history.append(test_acc)
            print(f"[Epoch {epoch+1}] Test Accuracy: {test_acc:.2f}% | Avg Train Loss: {avg_epoch_loss:.4f}")

    return {
        'model': model,
        'train_loss': train_loss_history,
        'test_acc': test_acc_history,
        'all_preds': all_preds,
        'all_labels': all_labels
    }