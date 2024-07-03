import torch
# Negative log-likelihood method

def mdn_loss(pi, sigma, mu, y):
    # This code creates a set of normal distribution parameterized by mu and sigma
    m = torch.distributions.Normal(loc=mu, scale=sigma) 

    # Transform y to have the shape (batch_size, num_gaussians)
    y = y.unsqueeze(1).expand(y.size(0), pi.size(1))  

    #Here we use a numerically stable computation method when calculating sum of exponential. 
    loss = torch.logsumexp(torch.log(pi) + m.log_prob(y), dim=1)  # shape (batch_size,)
    
    return -torch.mean(loss)

