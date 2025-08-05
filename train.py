import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import wandb
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

from model import ConditionalUNet, COLOR_MAP, get_color_idx

class PolygonDataset(Dataset):
    def __init__(self, data_json_path, input_dir, output_dir, transform=None):
        with open(data_json_path, 'r') as f:
            self.data = json.load(f)
        
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load input polygon image
        input_path = os.path.join(self.input_dir, item['input_polygon'])
        input_image = Image.open(input_path).convert('RGB')
        
        # Load output colored polygon image
        output_path = os.path.join(self.output_dir, item['output_image'])
        output_image = Image.open(output_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            input_image = self.transform(input_image)
            output_image = self.transform(output_image)
        
        # Get color index
        color_idx = get_color_idx(item['colour'])
        
        return {
            'input': input_image,
            'output': output_image,
            'color': torch.tensor(color_idx, dtype=torch.long),
            'color_name': item['colour']
        }

def create_data_loaders(train_json, val_json, train_input_dir, train_output_dir, 
                       val_input_dir, val_output_dir, batch_size=8, img_size=256):
    
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        # transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomRotation(degrees=15),
        transforms.ToTensor()
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])
    
    train_dataset = PolygonDataset(train_json, train_input_dir, train_output_dir, train_transform)
    val_dataset = PolygonDataset(val_json, val_input_dir, val_output_dir, val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def calculate_metrics(pred, target):
    """Calculate various metrics for evaluation"""
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    
    mse = mean_squared_error(target_np.flatten(), pred_np.flatten())
    
    # PSNR: Peak Signal-to-Noise Ratio
    # pixel value are normalised to [0, 1] => maxI =1
    psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')
    
    return {
        'mse': mse,
        'psnr': psnr
    }

def tensor_to_image(tensor):
    """
    Convert tensor to numpy array for visualization
    Handles proper format conversion and value clamping
    """
    # Move to CPU and detach from computation graph
    tensor = tensor.detach().cpu()
    
    # [C, H, W] to [H, W, C]
    if tensor.dim() == 3:
        tensor = tensor.permute(1, 2, 0)
    
    # Clamp values to [0, 1] range
    tensor = torch.clamp(tensor, 0, 1)
    
    return tensor.numpy()

def log_images_to_wandb(inputs, targets, predictions, color_names, epoch, phase="val", max_images=4):
    """
    Log images to wandb with proper formatting
    """
    batch_size = min(inputs.shape[0], max_images)
    
    # Convert tensors to proper format
    wandb_images = []
    
    for i in range(batch_size):
        # Convert individual images
        input_img = tensor_to_image(inputs[i])
        target_img = tensor_to_image(targets[i])
        pred_img = tensor_to_image(predictions[i])
        
        # Convert to uint8 format for wandb
        input_img = (input_img * 255).astype(np.uint8)
        target_img = (target_img * 255).astype(np.uint8)
        pred_img = (pred_img * 255).astype(np.uint8)
        
        color_name = color_names[i] if i < len(color_names) else "unknown"
        
        # Create wandb Image objects
        wandb_images.extend([
            wandb.Image(input_img, caption=f"Input_{i+1}"),
            wandb.Image(target_img, caption=f"Target_{i+1}_{color_name}"),
            wandb.Image(pred_img, caption=f"Prediction_{i+1}_{color_name}")
        ])
    wandb.log({
        f"{phase}_images": wandb_images,
        "epoch": epoch
    })

def visualize_predictions(model, val_loader, device, epoch, num_samples=4):
    """Create and log visualization of model predictions"""
    model.eval()
    
    # samples for visualization
    all_inputs = []
    all_targets = []
    all_predictions = []
    all_color_names = []
    
    with torch.no_grad():
        samples_collected = 0
        for batch in val_loader:
            if samples_collected >= num_samples:
                break
                
            inputs = batch['input'].to(device)
            targets = batch['output'].to(device)
            colors = batch['color'].to(device)
            color_names = batch['color_name']
            
            predictions = model(inputs, colors)
            
            # Take samples from this batch
            batch_samples = min(num_samples - samples_collected, inputs.shape[0])
            
            all_inputs.append(inputs[:batch_samples])
            all_targets.append(targets[:batch_samples])
            all_predictions.append(predictions[:batch_samples])
            all_color_names.extend(color_names[:batch_samples])
            
            samples_collected += batch_samples
    
    if all_inputs:
        # Concatenate all samples
        all_inputs = torch.cat(all_inputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        all_predictions = torch.cat(all_predictions, dim=0)
        
        # Log individual images to wandb
        log_images_to_wandb(all_inputs, all_targets, all_predictions, all_color_names, epoch)
        
        # Create matplotlib comparison grid
        fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(min(num_samples, all_inputs.shape[0])):
            # Convert tensors for matplotlib
            input_img = tensor_to_image(all_inputs[i])
            target_img = tensor_to_image(all_targets[i])
            pred_img = tensor_to_image(all_predictions[i])
            
            color_name = all_color_names[i] if i < len(all_color_names) else "unknown"
            
            # Plot images
            axes[i, 0].imshow(input_img)
            axes[i, 0].set_title(f'Input Polygon {i+1}')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(target_img)
            axes[i, 1].set_title(f'Target ({color_name})')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(pred_img)
            axes[i, 2].set_title(f'Prediction ({color_name})')
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        plt.suptitle(f'Epoch {epoch+1} - Validation Results', fontsize=16, y=1.02)
        
        # Log the comparison grid
        wandb.log({
            "comparison_grid": wandb.Image(fig),
            "epoch": epoch
        })
        
        plt.close(fig)

def log_training_batch(model, batch, epoch, batch_idx, phase="train"):
    """Log training batch images occasionally"""
    if batch_idx % 50 == 0:  # Log every 50 batches
        inputs = batch['input']
        targets = batch['output']
        colors = batch['color']
        color_names = batch['color_name']
        
        # Generate predictions
        model.eval()
        with torch.no_grad():
            predictions = model(inputs.to(next(model.parameters()).device), 
                              colors.to(next(model.parameters()).device))
        
        # Log first 2 images from batch
        log_images_to_wandb(
            inputs[:2].cpu(), 
            targets[:2].cpu(), 
            predictions[:2].cpu(), 
            color_names[:2], 
            epoch, 
            phase=f"{phase}_batch_{batch_idx}", 
            max_images=2
        )

def train_model(config):
    # Initialize wandb
    wandb.init(project="polygon-colorization", config=config)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_json=config['train_json'],
        val_json=config['val_json'],
        train_input_dir=config['train_input_dir'],
        train_output_dir=config['train_output_dir'],
        val_input_dir=config['val_input_dir'],
        val_output_dir=config['val_output_dir'],
        batch_size=config['batch_size'],
        img_size=config['img_size']
    )
    
    # Initialize model
    model = ConditionalUNet(
        n_channels=3, 
        n_classes=3, 
        num_colors=len(COLOR_MAP),
        bilinear=config['bilinear']
    ).to(device)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(config['num_epochs']):
        model.train()
        running_loss = 0.0
        running_mse = 0.0
        running_psnr = 0.0
        
        for batch in train_loader:
            inputs = batch['input'].to(device)
            targets = batch['output'].to(device)
            colors = batch['color'].to(device)

            optimizer.zero_grad()
            outputs = model(inputs, colors)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            metrics = calculate_metrics(outputs, targets)
            running_mse += metrics['mse']
            running_psnr += metrics['psnr']
        
        # Average
        train_loss = running_loss / len(train_loader)
        train_mse = running_mse / len(train_loader)
        train_psnr = running_psnr / len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_mse = 0.0
        val_psnr = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['input'].to(device)
                targets = batch['output'].to(device)
                colors = batch['color'].to(device)

                outputs = model(inputs, colors)
                loss = criterion(outputs, targets)

                val_loss += loss.item()
                metrics = calculate_metrics(outputs, targets)
                val_mse += metrics['mse']
                val_psnr += metrics['psnr']
        
        val_loss /= len(val_loader)
        val_mse /= len(val_loader)
        val_psnr /= len(val_loader)
        val_losses.append(val_loss)
        
        # Update LR scheduler
        scheduler.step(val_loss)
        
        # Log epoch-level metrics
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_mse': train_mse,
            'val_mse': val_mse,
            'train_psnr': train_psnr,
            'val_psnr': val_psnr,
            'learning_rate': optimizer.param_groups[0]['lr']
        })
        
        print(f"Epoch {epoch+1}/{config['num_epochs']}:")
        print(f"  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"  Train PSNR: {train_psnr:.2f}, Val PSNR: {val_psnr:.2f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config
            }, 'best_model.pth')
            print(f"  New best model saved at epoch {epoch+1}")
        
        # Log validation visuals every 5 epochs
        if (epoch + 1) % 5 == 0:
            try:
                visualize_predictions(model, val_loader, device, epoch)
            except Exception as e:
                print(f"Error in visualization: {e}")
    
    # Final visualization
    try:
        visualize_predictions(model, val_loader, device, config['num_epochs']-1)
    except Exception as e:
        print(f"Final visualization error: {e}")
    
    wandb.finish()
    return model, train_losses, val_losses

if __name__ == "__main__":
    config = {
        # Data paths
        'train_json': 'dataset/training/data.json',
        'val_json': 'dataset/validation/data.json',
        'train_input_dir': 'dataset/training/inputs',
        'train_output_dir': 'dataset/training/outputs',
        'val_input_dir': 'dataset/validation/inputs',
        'val_output_dir': 'dataset/validation/outputs',
        
        # hyperparameters
        'img_size': 256,
        'batch_size': 8,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'num_epochs': 100,
        'bilinear': True,
    }
    
    # Train the model
    model, train_losses, val_losses = train_model(config)
    
    print("Training completed!")
