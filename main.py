import argparse
import torch
import torch.nn as nn
from models.model import GuavaClassifier
from src.data_preprocessing import load_data
from src.utils import save_model, get_device, AverageMeter
from src.optim.adopt import ADOPT
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm

def train(config):
    # Load data
    train_loader, val_loader, _ = load_data(config)

    # Initialize model with pretrained weights
    model = GuavaClassifier(num_classes=config['model']['num_classes'])
    device = get_device()
    model = model.to(device)

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Initialize custom optimizer with only trainable parameters
    optimizer = ADOPT(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['training']['learning_rate']
    )

    # Set the number of epochs
    epochs = config['training']['epochs']
    best_val_accuracy = 0.0

    # Initialize meters for loss and accuracies
    train_loss_meter = AverageMeter()
    val_loss_meter = AverageMeter()

    # Create results directory
    os.makedirs(config['training']['results_dir'], exist_ok=True)

    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss_meter.reset()
        correct_train = 0
        total_train = 0

        # Training phase with tqdm progress bar
        for inputs, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{epochs}", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Calculate Top-1 accuracy
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

            # Update train loss meter
            train_loss_meter.update(loss.item(), inputs.size(0))

        # Calculate and print training Top-1 accuracy
        train_top1_acc = correct_train / total_train * 100

        # Validation phase with tqdm progress bar
        model.eval()
        val_loss_meter.reset()
        correct_val = 0
        total_val = 0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Validating Epoch {epoch+1}/{epochs}", leave=False):
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Calculate Top-1 accuracy
                _, preds = torch.max(outputs, 1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

                # Update validation loss meter
                val_loss_meter.update(loss.item(), inputs.size(0))

                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        # Calculate and print validation Top-1 accuracy
        val_top1_acc = correct_val / total_val * 100

        # Flatten predictions and labels for confusion matrix and classification report
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        print(f"\nEpoch [{epoch+1}/{epochs}]")
        print(f"Train Loss: {train_loss_meter.avg:.4f}, "
              f"Train Top-1 Acc: {train_top1_acc:.4f}")
        print(f"Validation Loss: {val_loss_meter.avg:.4f}, "
              f"Validation Top-1 Acc: {val_top1_acc:.4f}")

        # Save classification results for validation
        cm = confusion_matrix(all_labels, all_preds)
        class_report = classification_report(all_labels, all_preds, target_names=config['data']['class_names'])
        with open(os.path.join(config['training']['results_dir'], f'validation_epoch_{epoch+1}_report.txt'), 'w') as f:
            f.write(f"Confusion Matrix:\n{cm}\n\n")
            f.write(f"Classification Report:\n{class_report}")

        # Plot and save confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=config['data']['class_names'],
                    yticklabels=config['data']['class_names'])
        plt.title(f'Confusion Matrix - Epoch {epoch+1}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(os.path.join(config['training']['results_dir'], f'confusion_matrix_epoch_{epoch+1}.png'))
        plt.close()

        # Check if current validation accuracy is the best
        if val_top1_acc > best_val_accuracy:
            best_val_accuracy = val_top1_acc
            save_model(model, config['training']['save_path'])
            print("Best model saved!")

    print("\nTraining complete.")
    print(f"Best Validation Top-1 Accuracy: {best_val_accuracy:.4f}")

def evaluate(config):
    # Load data
    _, _, test_loader = load_data(config)

    # Initialize model with pretrained weights
    model = GuavaClassifier(num_classes=config['model']['num_classes'])
    device = get_device()
    model = model.to(device)

    # Load the best model with map_location
    model.load_state_dict(torch.load(config['training']['save_path'], map_location=device, weights_only=True))
    model.eval()

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Test phase
    test_loss_meter = AverageMeter()
    correct_test = 0
    total_test = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating Test Set", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            # Calculate loss
            loss = criterion(outputs, labels)

            # Calculate Top-1 accuracy
            _, preds = torch.max(outputs, 1)
            correct_test += (preds == labels).sum().item()
            total_test += labels.size(0)

            # Update loss meter
            test_loss_meter.update(loss.item(), inputs.size(0))

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # Calculate final accuracy on test set
    test_top1_acc = correct_test / total_test * 100  # Correcting accuracy computation

    print(f"\nTest Top-1 Accuracy: {test_top1_acc:.4f}")

    # Flatten the predictions and labels
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)  # Using top-1 predictions
    print(f"Confusion Matrix:\n{cm}")

    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=config['data']['class_names'],
                yticklabels=config['data']['class_names'])
    plt.title('Confusion Matrix - Test Set')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(config['training']['results_dir'], 'confusion_matrix_test.png'))
    plt.close()

    # Classification Report
    class_report = classification_report(all_labels, all_preds, target_names=config['data']['class_names'])
    print(f"Classification Report:\n{class_report}")

    # Save classification report
    with open(os.path.join(config['training']['results_dir'], 'classification_report_test.txt'), 'w') as f:
        f.write(f"Confusion Matrix:\n{cm}\n\n")
        f.write(f"Classification Report:\n{class_report}")

    print("Evaluation complete.")

def main():
    parser = argparse.ArgumentParser(description='Train or evaluate the model.')
    parser.add_argument('--mode', choices=['train', 'evaluate'], required=True, help='Mode: train or evaluate')
    args = parser.parse_args()

    # Configuration parameters
    config = {
        'data': {
            'train_dir': './data/GuavaDiseaseDataset/GuavaDiseaseDataset/train',
            'val_dir': './data/GuavaDiseaseDataset/GuavaDiseaseDataset/val',
            'test_dir': './data/GuavaDiseaseDataset/GuavaDiseaseDataset/test',
            'batch_size': 4,
            'class_names': [
                'Anthracnose',
                'Fruit Fly',
                'Healthy Guava'
            ]
        },
        'model': {
            'num_classes': 3
        },
        'training': {
            'epochs': 15,
            'learning_rate': 0.001,
            'save_path': './models/best_M11217073.pth',
            'results_dir': './results'
        }
    }

    if args.mode == 'train':
        train(config)
    elif args.mode == 'evaluate':
        evaluate(config)

if __name__ == "__main__":
    main()