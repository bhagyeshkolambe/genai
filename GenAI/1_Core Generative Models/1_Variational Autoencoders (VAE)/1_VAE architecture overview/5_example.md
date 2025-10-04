Below is an implementation of a basic Variational Autoencoder (VAE) using PyTorch, along with easier explanations.

```python
import torch
from torch import nn
import torch.nn.functional as F

# Define the VAE class
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        
        # Encoder part
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * latent_dim)  # Output is mean and log variance for each dimension of the latent space
        )
        
        # Decoder part
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def encode(self, x):
        """Encode input into mean and log variance."""
        output = self.encoder(x)  # Pass through the encoder network
        mu, log_var = torch.chunk(output, chunks=2, dim=-1)  # Split the output into two parts: mean and log variance
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        """Reparametrization trick to sample from a Gaussian distribution."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)  # Sample random noise from a standard normal distribution
        return mu + eps * std
    
    def decode(self, z):
        """Decode latent vector back into the original input space."""
        return self.decoder(z)
    
    def forward(self, x):
        """Forward pass through the VAE."""
        mu, log_var = self.encode(x)  # Get mean and log variance from the encoder
        z = self.reparameterize(mu, log_var)  # Sample a latent vector using reparametrization trick
        recon_x = self.decode(z)  # Decode the sampled latent vector back into input space
        
        return recon_x, mu, log_var

# Example usage:
if __name__ == "__main__":
    # Define parameters
    input_dim = 784  # For example, images are flattened into a single row of 784 pixels (28x28)
    hidden_dim = 512  # Size of the hidden layer in both encoder and decoder
    latent_dim = 20   # Dimensionality of the latent space
    
    vae = VAE(input_dim, hidden_dim, latent_dim)  # Instantiate the VAE model

    # Dummy input data (e.g., a batch of images)
    x = torch.randn(10, input_dim)  # Batch size is 10, each sample has input_dim features
    
    recon_x, mu, log_var = vae(x)  # Perform forward pass through the VAE

    print("Reconstructed Input Shape:", recon_x.shape)  # Should be (10, 784)
    print("Mean of Latent Space Dimensions:", mu.shape)  # Should be (10, 20)
    print("Log Variance of Latent Space Dimensions:", log_var.shape)  # Should also be (10, 20)
```

### Explanation:

1. **VAE Class Definition**:
   - The `VAE` class inherits from `nn.Module`, making it a PyTorch module.
   - It contains three main parts: Encoder, Decoder, and the overall Forward Pass.

2. **Encoder**:
   - Takes input data (`x`) and passes it through two fully connected layers followed by ReLU activation functions.
   - Outputs are split into mean (`mu`) and log variance (`log_var`) for each dimension of the latent space.

3. **Reparametrization Trick**:
   - Samples a random noise vector `eps` from a standard normal distribution (mean 0, variance 1).
   - Combines this with the mean and standard deviation derived from the encoder output to generate samples from a Gaussian distribution.

4. **Decoder**:
   - Takes the sampled latent vector (`z`) and passes it through two fully connected layers followed by ReLU activation functions.
   - Outputs are reshaped to match the original input dimensions.

5. **Forward Pass**:
   - Encodes the input data, samples from the Gaussian distribution defined by the encoder's output, decodes this sample back into the input space, and returns these results along with intermediate values for further analysis.

This setup allows you to train a VAE on your dataset, where the latent space captures meaningful representations of the input data while enabling generation of new data instances.