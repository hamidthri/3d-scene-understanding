from config import get_args
from data_loader import get_data_loaders
from model_factory import create_model
from trainer import Trainer
from utils.visualization import plot_training_curves

def main():
    args = get_args()
    
    train_loader, test_loader, classes = get_data_loaders(args)
    
    model = create_model(args.model, len(classes), args)
    
    trainer = Trainer(model, train_loader, test_loader, args, classes)
    
    train_losses, train_accuracies, test_accuracies = trainer.train()
    
    plot_training_curves(train_losses, train_accuracies, test_accuracies)

if __name__ == "__main__":
    main()