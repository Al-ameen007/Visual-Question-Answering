import torch
from tqdm import tqdm
from pathlib import Path
from transformers import BlipForQuestionAnswering, AutoProcessor
from data import create_dataloaders


def prepare_model(model_name='Salesforce/blip-vqa-base', device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = BlipForQuestionAnswering.from_pretrained(model_name)
    processor = AutoProcessor.from_pretrained(model_name)
    
    model.to(device)
    
    return model, processor, device


def save_model(model, processor, save_dir, epoch=None):
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    if epoch is not None:
        save_path = save_path / f"checkpoint-epoch-{epoch}"
        save_path.mkdir(parents=True, exist_ok=True)
    
    model.save_pretrained(save_path)
    processor.save_pretrained(save_path)
    
    print(f"Model saved to {save_path}")
    return str(save_path)


def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, save_dir):
    checkpoint_path = Path(save_dir) / f"checkpoint-epoch-{epoch}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
    }, checkpoint_path)
    
    print(f"Checkpoint saved to {checkpoint_path}")
    return str(checkpoint_path)


def calc_loss_batch(batch, model, device):
    batch = {k: v.to(device) for k, v in batch.items()}
    outputs = model(**batch)
    return outputs.loss


def calc_loss_loader(dataloader, model, device, num_batches=None):
    total_loss = 0
    
    if len(dataloader) == 0:
        return float('nan')
    elif num_batches is None:
        num_batches = len(dataloader)
    else:
        num_batches = min(num_batches, len(dataloader))
    
    for i, batch in enumerate(dataloader):
        if i < num_batches:
            loss = calc_loss_batch(batch, model, device)
            total_loss += loss.item()
        else:
            break
    
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, eval_iter)
    
    model.train()
    return train_loss, val_loss


def train_model(
    train_url,
    val_url,
    model_name='Salesforce/blip-vqa-base',
    data_dir='./data',
    save_dir='./checkpoints',
    num_epochs=3,
    batch_size=8,
    learning_rate=5e-5,
    weight_decay=0.01,
    num_workers=2,
    eval_freq=100,
    eval_iter=50,
    save_every_epoch=True
):
    print("="*60)
    print("Starting Training")
    print("="*60)
    
    model, processor, device = prepare_model(model_name)
    
    print("\nLoading data...")
    train_loader, val_loader = create_dataloaders(
        train_url=train_url,
        val_url=val_url,
        processor_name_or_path=model_name,
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    model.train()
    train_losses, val_losses, track_steps = [], [], []
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print('='*60)
        
        model.train()
        epoch_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            optimizer.zero_grad()
            loss = calc_loss_batch(batch, model, device)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            global_step += 1
            
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                track_steps.append(global_step)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                
                print(f"\nStep {global_step}:")
                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Val Loss: {val_loss:.4f}")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_model(model, processor, f"{save_dir}/best_model")
                    print(f"  New best model saved! (Val Loss: {val_loss:.4f})")
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"\nEpoch {epoch + 1} Average Loss: {avg_epoch_loss:.4f}")
        
        train_loss, val_loss = evaluate_model(
            model, train_loader, val_loader, device, eval_iter=None
        )
        print(f"Epoch {epoch + 1} Evaluation:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        
        if save_every_epoch:
            save_checkpoint(model, optimizer, epoch + 1, train_loss, val_loss, save_dir)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    
    final_save_path = save_model(model, processor, f"{save_dir}/final_model")
    print(f"\nFinal model saved to: {final_save_path}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    return model, processor, train_losses, val_losses, track_steps