import torch
from config import patience, delta

# Negative log-likelihood method
def mdn_loss(pi, sigma, mu, y):
    # This code creates a set of normal distribution parameterized by mu and sigma
    m = torch.distributions.Normal(loc=mu, scale=sigma) 

    # Transform y to have the shape (batch_size, num_gaussians)
    y = y.unsqueeze(1).expand(y.size(0), pi.size(1))  

    #Here we use a numerically stable computation method when calculating sum of exponential. 
    loss = torch.logsumexp(torch.log(pi) + m.log_prob(y), dim=1)  # shape (batch_size,)
    
    return -torch.mean(loss)

# Early stopping class, which monitors validation loss to prevent overfitting
class EarlyStopping:
    def __init__(self, patience=patience, verbose=False, delta=delta):
        """
        patience: Number of consecutive epochs without improvement in validation loss allowed.
        verbose: if True, prints early stopping information
        delta: Minimum change in validation loss to be considered significant improvenment
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_loss = None
        self.epochs_no_improve = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss + self.delta:
            self.epochs_no_improve += 1
            if self.epochs_no_improve >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("Early stopping triggered")
        else:
            self.best_loss = val_loss
            self.epochs_no_improve = 0