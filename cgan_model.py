import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Check if multiple GPUs are available
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    multi_gpu = True
else:
    multi_gpu = False

# Hyperparameters
latent_dim = 100
num_classes = 10
batch_size = 128
epochs = 100
lr = 0.0002
beta1 = 0.5
beta2 = 0.999
image_size = 28

# Data loading
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])

# Download and load the MNIST dataset
train_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

# Generator Network
class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super(Generator, self).__init__()
        
        self.label_embedding = nn.Embedding(num_classes, num_classes)
        
        # Input layers for noise and label
        self.input_layer = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Hidden layers
        self.hidden_layers = nn.Sequential(
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(1024, image_size * image_size),
            nn.Tanh()  # Output range [-1, 1]
        )
        
    def forward(self, noise, labels):
        # Embed the labels
        label_embedding = self.label_embedding(labels)
        
        # Concatenate noise and label embedding
        x = torch.cat((noise, label_embedding), dim=1)
        
        # Forward pass through network
        x = self.input_layer(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        
        # Reshape to image dimensions
        x = x.view(x.size(0), 1, image_size, image_size)
        
        return x

# Discriminator Network
class Discriminator(nn.Module):
    def __init__(self, num_classes):
        super(Discriminator, self).__init__()
        
        # Label embedding
        self.label_embedding = nn.Embedding(num_classes, num_classes)
        
        # Input layer for flattened image and label
        self.input_layer = nn.Sequential(
            nn.Linear(image_size * image_size + num_classes, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3)
        )
        
        # Hidden layers
        self.hidden_layers = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
        )
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid()  # Output probability [0, 1]
        )
    
    def forward(self, img, labels):
        # Flatten the image
        img_flat = img.view(img.size(0), -1)
        
        # Embed the labels
        label_embedding = self.label_embedding(labels)
        
        # Concatenate image and label embedding
        x = torch.cat((img_flat, label_embedding), dim=1)
        
        # Forward pass through network
        x = self.input_layer(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        
        return x

# Initialize models
generator = Generator(latent_dim, num_classes)
discriminator = Discriminator(num_classes)

# Move models to device
generator.to(device)
discriminator.to(device)

# Use DataParallel if multiple GPUs are available
if multi_gpu:
    generator = nn.DataParallel(generator)
    discriminator = nn.DataParallel(discriminator)

# Loss function
adversarial_loss = nn.BCELoss()

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))

# Training function
def train_cgan(epochs, save_interval=10):
    # Create directories for saving samples and models if they don't exist
    os.makedirs("images", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Lists to keep track of losses
    G_losses = []
    D_losses = []
    
    print("Starting training...")
    
    for epoch in range(epochs):
        for i, (real_images, labels) in enumerate(train_loader):
            # Move data to device
            real_images = real_images.to(device)
            labels = labels.to(device)
            
            # Create labels for real and fake images
            real_target = torch.ones(real_images.size(0), 1).to(device)
            fake_target = torch.zeros(real_images.size(0), 1).to(device)
            
            #------------------------------------------------------
            # Train Discriminator
            #------------------------------------------------------
            optimizer_D.zero_grad()
            
            # Discriminator loss on real images
            real_pred = discriminator(real_images, labels)
            d_real_loss = adversarial_loss(real_pred, real_target)
            
            # Generate fake images
            noise = torch.randn(real_images.size(0), latent_dim).to(device)
            fake_images = generator(noise, labels)
            
            # Discriminator loss on fake images
            fake_pred = discriminator(fake_images.detach(), labels)
            d_fake_loss = adversarial_loss(fake_pred, fake_target)
            
            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()
            
            #------------------------------------------------------
            # Train Generator
            #------------------------------------------------------
            optimizer_G.zero_grad()
            
            # Generator tries to fool discriminator
            fake_pred = discriminator(fake_images, labels)
            g_loss = adversarial_loss(fake_pred, real_target)
            
            g_loss.backward()
            optimizer_G.step()
            
            # Print training stats
            if i % 100 == 0:
                print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(train_loader)}] "
                      f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")
            
            # Save losses for plotting
            G_losses.append(g_loss.item())
            D_losses.append(d_loss.item())
        
        # Save model checkpoints
        if epoch % save_interval == 0 or epoch == epochs - 1:
            torch.save(generator.state_dict(), f"models/generator_epoch_{epoch}.pth")
            torch.save(discriminator.state_dict(), f"models/discriminator_epoch_{epoch}.pth")
            
            # Generate and save sample images
            generate_samples(epoch)
    
    # Plot loss curves
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss")
    plt.plot(G_losses, label="Generator")
    plt.plot(D_losses, label="Discriminator")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("training_loss.png")
    plt.close()
    
    print("Training complete!")
    return generator, discriminator

# Function to generate sample images
def generate_samples(epoch, nrows=5, ncols=10):
    """Generate and save sample images for each digit class"""
    generator.eval()
    
    with torch.no_grad():
        # Create fixed noise
        noise = torch.randn(ncols * nrows, latent_dim).to(device)
        
        # Generate images for each class
        fig, axs = plt.subplots(nrows, ncols, figsize=(12, 6))
        
        for digit in range(nrows):
            # Create labels for specific digit
            labels = torch.full((ncols,), digit, dtype=torch.long).to(device)
            
            # Generate images
            fake_images = generator(noise[digit*ncols:(digit+1)*ncols], labels)
            fake_images = fake_images.cpu().detach()
            
            # Denormalize images
            fake_images = fake_images * 0.5 + 0.5
            
            # Plot images
            for i in range(ncols):
                axs[digit, i].imshow(fake_images[i].squeeze(), cmap='gray')
                axs[digit, i].axis('off')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(f"images/samples_epoch_{epoch}.png")
        plt.close()
    
    generator.train()

# Function to generate specific digit images for inference
def generate_digit_samples(digit, num_samples=5, model_path=None):
    """Generate images for a specific digit"""
    # Load trained model if path is provided
    if model_path is not None:
        state_dict = torch.load(model_path, map_location=device)
        generator.load_state_dict(state_dict)
    
    generator.eval()
    
    with torch.no_grad():
        # Create random noise
        noise = torch.randn(num_samples, latent_dim).to(device)
        
        # Create labels for the digit
        labels = torch.full((num_samples,), digit, dtype=torch.long).to(device)
        
        # Generate images
        fake_images = generator(noise, labels)
        
        # Denormalize
        fake_images = fake_images * 0.5 + 0.5
        
        return fake_images.cpu()

# Main execution
if __name__ == "__main__":
    print("MNIST Conditional GAN Training")
    
    # Train the model
    # Uncomment to train:
    # generator, discriminator = train_cgan(epochs=epochs)
    
    # Or load a pre-trained model for inference:
    # generator.load_state_dict(torch.load("models/generator_final.pth", map_location=device))
    
    print("Model ready for inference")