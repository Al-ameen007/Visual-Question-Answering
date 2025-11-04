import argparse
import matplotlib.pyplot as plt
from engine import train_model


def parse_args():
    parser = argparse.ArgumentParser(description='Train BLIP-VQA model on VizWiz dataset')
    
    parser.add_argument('--train_url', type=str, 
                       default='https://vizwiz.cs.colorado.edu/VizWiz_final/images/train.zip',
                       help='URL to training data zip file')
    parser.add_argument('--val_url', type=str,
                       default='https://vizwiz.cs.colorado.edu/VizWiz_final/images/val.zip',
                       help='URL to validation data zip file')
    parser.add_argument('--model_name', type=str,
                       default='Salesforce/blip-vqa-base',
                       help='Pretrained model name or path')
    parser.add_argument('--data_dir', type=str,
                       default='./data',
                       help='Directory to store downloaded data')
    parser.add_argument('--save_dir', type=str,
                       default='./checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--num_epochs', type=int,
                       default=3,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int,
                       default=16,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float,
                       default=5e-5,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float,
                       default=0.01,
                       help='Weight decay for AdamW optimizer')
    parser.add_argument('--num_workers', type=int,
                       default=4,
                       help='Number of workers for data loading')
    parser.add_argument('--eval_freq', type=int,
                       default=100,
                       help='Evaluate every N steps')
    parser.add_argument('--eval_iter', type=int,
                       default=50,
                       help='Number of batches to use for evaluation')
    parser.add_argument('--save_every_epoch', action='store_true',
                       help='Save checkpoint after every epoch')
    parser.add_argument('--plot_training', action='store_true',
                       help='Plot and save training curve')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("Training Configuration:")
    print("-" * 60)
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("-" * 60)
    
    model, processor, train_losses, val_losses, steps = train_model(
        train_url=args.train_url,
        val_url=args.val_url,
        model_name=args.model_name,
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        eval_freq=args.eval_freq,
        eval_iter=args.eval_iter,
        save_every_epoch=args.save_every_epoch
    )
    
    if args.plot_training:
        plt.figure(figsize=(10, 5))
        plt.plot(steps, train_losses, label='Train Loss')
        plt.plot(steps, val_losses, label='Validation Loss')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training Progress')
        plt.savefig('training_curve.png')
        print("\nTraining curve saved to training_curve.png")
    
    print("\nTraining complete!")
    print(f"Model saved in {args.save_dir}/final_model")
    print(f"Best model saved in {args.save_dir}/best_model")


if __name__ == '__main__':
    main()