import torch
from config import patience, delta, entropy_weight

# Negative log-likelihood method with Entropy regularization weight
def mdn_loss(pi, sigma, mu, y):
    # Ensure sigma is within a reasonable range to prevent division by extremely small numbers
    sigma = torch.clamp(sigma, min=1e-6)
    
    # Create a set of normal distributions parameterized by mu and sigma
    m = torch.distributions.Normal(loc=mu, scale=sigma)

    # Transform y to have the shape (batch_size, num_gaussians)
    y = y.unsqueeze(1).expand_as(pi)

    # Calculate the log probabilities for each Gaussian
    log_prob = m.log_prob(y)

    # Using a numerically stable method to calculate the sum of exponentials for the mixture
    log_pi = torch.log(torch.clamp(pi, min=1e-6))  # Clamp pi before taking log to avoid log(0)
    log_sum_exp = torch.logsumexp(log_pi + log_prob, dim=1)  # Log-sum-exp for mixture
    
    # Compute the negative log likelihood
    nll_loss = -torch.mean(log_sum_exp)

    # Compute the entropy of the mixing coefficients pi
    epsilon = 1e-5  # Small constant to ensure numerical stability
    pi_clamped = torch.clamp(pi, min=epsilon, max=1-epsilon)  # Clamp pi to avoid log(0)
    entropy = -torch.sum(pi_clamped * torch.log(pi_clamped), dim=1).mean()

    # Calculate total loss, considering the entropy regularization
    total_loss = nll_loss - entropy_weight * entropy

    return total_loss


# Negative log-likelihood method
def mdn_loss_test(pi, sigma, mu, y):
    # Avoid very small sigma values
    sigma = torch.clamp(sigma, min=1e-6)

    # This code creates a set of normal distribution parameterized by mu and sigma
    m = torch.distributions.Normal(loc=mu, scale=sigma) 

    # Transform y to have the shape (batch_size, num_gaussians)
    y = y.unsqueeze(1).expand(y.size(0), pi.size(1))  

    # Add a small constant to pi before taking log to avoid log(0)
    # Here we use a numerically stable computation method when calculating sum of exponential.
    log_pi = torch.log(pi + 1e-6)
    log_prob = m.log_prob(y)  # log probability of y given the distribution

    # Calculate the log-sum-exp of the log probabilities and log mixing coefficients for numerical stability
    loss = torch.logsumexp(log_pi + log_prob, dim=1)  # shape (batch_size,)

    # Return the negative mean of these log probabilities
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