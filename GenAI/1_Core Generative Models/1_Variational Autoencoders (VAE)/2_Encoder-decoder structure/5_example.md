The main function is to generate image, but you must explain how it works.
Here's an implementation of a simple Encoder-Decoder structure in Variational Autoencoder (VAE) using PyTorch:

```python
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Define the device we will use for training and inference
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims=[256], latent_dim=10):
        super(Encoder, self).__init__()

        # Define the layers for encoding input to latent space
        encoders = []
        in_channels = input_dim  # Input channels (e.g., RGB images have 3 channels)
        
        # Create a list of dimensions by iterating over hidden_dims and append latent_dim at last position.
        dims = [in_channels] + hidden_dims + [latent_dim*2]
        
        # Iterate through the dimensions and create fully connected layers
        for i in range(len(dims)-1):
            encoders.append(torch.nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                encoders.append(torch.nn.ReLU())
                
        self.encoders = torch.nn.Sequential(*encoders)

    def forward(self, x):
        return self.encoders(x)


class Decoder(torch.nn.Module):
    def __init__(self, latent_dim=10, hidden_dims=[256], output_dim=(3, 28, 28)):
        super(Decoder, self).__init__()

        # Define the layers for decoding from latent space to input space
        decoders = []
        out_channels = int(torch.prod(torch.tensor(output_dim)))  # Calculate total number of pixels in image
        
        dims = [latent_dim] + hidden_dims[::-1] + [out_channels]
        
        # Iterate through the dimensions and create fully connected layers
        for i in range(len(dims)-1):
            decoders.append(torch.nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                decoders.append(torch.nn.ReLU())
                
        self.decoders = torch.nn.Sequential(*decoders)

    def forward(self, z):
        # Reshape output of decoder to match original input shape
        return self.decoders(z).view(-1, *output_dim)


class VAE(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def forward(self, x):
        # Encoding step: pass input through the encoder network
        z_mu, z_logvar = self.encoder(x).chunk(2, dim=1)
        
        # Reparameterization trick to sample from N(mu, var) distribution
        z_sampled = self.reparameterize(z_mu, z_logvar)

        # Decoding step: pass sampled latent variable through the decoder network
        recon_x = self.decoder(z_sampled)
    
        return recon_x, z_mu, z_logvar


# Load CIFAR-10 dataset and apply transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Create data loaders for training and testing
batch_size = 64
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)


input_dim = (3, 28, 28)  # Assuming input images are resized to 28x28 pixels
latent_dim = 10

# Instantiate encoder and decoder models
encoder_model = Encoder(input_dim=input_dim, latent_dim=latent_dim).to(device)
decoder_model = Decoder(latent_dim=latent_dim, output_dim=(3, 28, 28)).to(device)

vae_model = VAE(encoder=encoder_model, decoder=decoder_model).to(device)


# Define loss function and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(vae_model.parameters(), lr=1e-3)


def train(model, loader):
    model.train()
    
    total_loss = 0
    
    for inputs, _ in loader:
        inputs = inputs.to(device)
        
        recon_x, z_mu, z_logvar = model(inputs)
        
        loss_recon = criterion(recon_x.view(-1, torch.prod(torch.tensor(input_dim))), 
                               inputs.view(-1, torch.prod(torch.tensor(input_dim))))
        
        kl_divergence = -0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_logvar.exp())
        
        loss = loss_recon + kl_divergence
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * inputs.size(0)
    
    return total_loss / len(loader.dataset)


def test(model, loader):
    model.eval()
    
    total_loss = 0
    
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)

            recon_x, z_mu, z_logvar = model(inputs)

            loss_recon = criterion(recon_x.view(-1, torch.prod(torch.tensor(input_dim))), 
                                   inputs.view(-1, torch.prod(torch.tensor(input_dim))))
            
            kl_divergence = -0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_logvar.exp())
                
            loss = loss_recon + kl_divergence
            
            total_loss += loss.item() * inputs.size(0)
        
        return total_loss / len(loader.dataset)


# Train and evaluate the VAE model
num_epochs = 10

train_losses, test_losses = [], []

for epoch in range(num_epochs):
    train_loss = train(vae_model, train_loader)
    test_loss = test(vae_model, test_loader)

    print(f'Epoch [{epoch+1}/{num_epochs}] | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}')

    train_losses.append(train_loss)
    test_losses.append(test_loss)


# Visualize results
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

```