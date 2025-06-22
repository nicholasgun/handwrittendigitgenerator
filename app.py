import streamlit as st
import torch
import numpy as np
from PIL import Image
import io
import os
from cgan_model import Generator, latent_dim

# Page configuration
st.set_page_config(page_title="Handwritten Digit Generator", layout="centered")

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Function to load model
@st.cache_resource
def load_model(model_path):
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Model parameters
    num_classes = 10
    
    # Initialize model
    generator = Generator(latent_dim, num_classes)
    
    # Load model weights
    try:
        state_dict = torch.load(model_path, map_location=device)
        
        # Handle DataParallel model state_dict - remove 'module.' prefix if it exists
        if all(k.startswith('module.') for k in state_dict.keys()):
            # Create new OrderedDict without the 'module.' prefix
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove 'module.' prefix
                new_state_dict[name] = v
            state_dict = new_state_dict
                
        generator.load_state_dict(state_dict)
        generator.to(device)
        generator.eval()
        st.session_state["model_loaded"] = True
        return generator, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.session_state["model_loaded"] = False
        return None, device

# Function to generate and display images
def generate_images(digit, generator, device):
    if not st.session_state.get("model_loaded", False):
        st.warning("Model is not loaded properly. Please check the model path.")
        return
        
    try:
        # Create random noise
        noise = torch.randn(5, latent_dim).to(device)
        
        # Create labels for the digit
        labels = torch.full((5,), digit, dtype=torch.long).to(device)
        
        # Generate images
        with torch.no_grad():
            fake_images = generator(noise, labels)
            
            # Denormalize
            fake_images = fake_images * 0.5 + 0.5
            
        # Display the images
        st.subheader(f"Generated images of digit {digit}")
        
        cols = st.columns(5)
        for i, col in enumerate(cols):
            # Convert tensor to PIL Image
            img = fake_images[i].cpu().squeeze().numpy()
            img = (img * 255).astype(np.uint8)
            img_pil = Image.fromarray(img)
            
            # Display in column
            with col:
                st.image(img_pil, caption=f"Sample {i+1}", use_container_width=True)
                
    except Exception as e:
        st.error(f"Error generating images: {e}")

# Main application
def main():
    # Title and description
    st.markdown("<h1 style='text-align: center;'>Handwritten Digit Image Generator</h1>", unsafe_allow_html=True)
    st.markdown("Generate synthetic MNIST-like images using your trained model.")
    
    # Model path - using absolute path for reliability
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "models", "generator_epoch_99.pth")
    
    # Check if model exists
    if not os.path.exists(model_path):
        st.warning(f"Model file not found at {model_path}. Please make sure you've trained the model and specified the correct path.")
    
    # Load model
    generator, device = load_model(model_path)
    
    # User input
    # Using the selectbox label instead of a separate markdown header
    
    # Digit selection with proper label for accessibility
    digit = st.selectbox("Choose a digit to generate (0-9):", options=list(range(10)), index=2)
    
    # Generate button
    if st.button("Generate Images", type="primary", use_container_width=False):
        generate_images(digit, generator, device)

if __name__ == "__main__":
    main()
