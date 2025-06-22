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
    
    # Check if model file exists
    if not os.path.exists(model_path):
        return None, device, f"Model file not found at: {model_path}"
    
    # Initialize model
    generator = Generator(latent_dim, num_classes)
    
    # Load model weights
    try:
        state_dict = torch.load(model_path, map_location=device)
        
        # Handle DataParallel model state_dict - remove 'module.' prefix if it exists
        if state_dict and all(k.startswith('module.') for k in state_dict.keys()):
            # Create new OrderedDict without the 'module.' prefix
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove 'module.' prefix
                new_state_dict[name] = v
            state_dict = new_state_dict
        
        # Load state dict into model
        generator.load_state_dict(state_dict)
        generator.to(device)
        generator.eval()
        
        return generator, device, "success"
        
    except Exception as e:
        return None, device, f"Error loading model: {str(e)}"

# Function to generate and display images
def generate_images(digit, generator, device):
    # Double-check that model is loaded properly
    if not st.session_state.get("model_loaded", False) or generator is None:
        st.error("❌ Model is not loaded properly. Please check the model path and try reloading the page.")
        return
        
    try:
        with st.spinner(f"Generating 5 images of digit {digit}..."):
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
                
                # Ensure values are in valid range
                img = np.clip(img, 0, 1)
                img = (img * 255).astype(np.uint8)
                img_pil = Image.fromarray(img)
                
                # Display in column
                with col:
                    st.image(img_pil, caption=f"Sample {i+1}", use_container_width=True)
                
    except RuntimeError as e:
        st.error(f"❌ Runtime error during generation: {str(e)}")
        st.info("This might be a memory or device issue. Try refreshing the page.")
    except Exception as e:
        st.error(f"❌ Error generating images: {str(e)}")
        st.info(f"Error type: {type(e).__name__}")
        st.info("Please try again or refresh the page.")

# Main application
def main():
    # Title and description
    st.markdown("<h1 style='text-align: center;'>Handwritten Digit Image Generator</h1>", unsafe_allow_html=True)
    st.markdown("Generate synthetic MNIST-like images using your trained model.")
    
    # Model path - using absolute path for reliability
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, "models")
    
    # Check if models directory exists
    if not os.path.exists(models_dir):
        st.error(f"Models directory not found at: {models_dir}")
        st.info("Please make sure the 'models' directory exists with your trained model file.")
        return
    
    # List available model files
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
    
    if not model_files:
        st.error("No model files (.pth) found in the models directory.")
        st.info(f"Please upload your trained model file to: {models_dir}")
        return
    
    # If multiple models exist, let user choose (for debugging purposes)
    if len(model_files) > 1:
        st.info(f"Found {len(model_files)} model files: {', '.join(model_files)}")
        # For now, we'll use the first one, but you could add a selectbox here
        selected_model = model_files[0]
        st.info(f"Using model: {selected_model}")
    else:
        selected_model = model_files[0]
    
    model_path = os.path.join(models_dir, selected_model)
    
    # Load model with improved error handling
    generator, device, status = load_model(model_path)
    
    # Handle the loading result and set session state
    if status == "success" and generator is not None:
        st.session_state["model_loaded"] = True
        model_loaded = True
    else:
        st.session_state["model_loaded"] = False
        model_loaded = False
        if status != "success":
            st.error(f"❌ {status}")
    
    # Only show the interface if model loaded successfully
    if model_loaded and generator is not None:
        # User input
        # Digit selection with proper label for accessibility
        digit = st.selectbox("Choose a digit to generate (0-9):", options=list(range(10)), index=2)
        
        # Generate button
        if st.button("Generate Images", type="primary", use_container_width=False):
            generate_images(digit, generator, device)
    else:
        st.warning("⚠️ Please fix the model loading issues above before proceeding.")

if __name__ == "__main__":
    main()
