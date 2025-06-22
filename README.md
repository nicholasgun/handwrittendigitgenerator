# Handwritten Digit Image Generator

This project uses a Conditional Generative Adversarial Network (CGAN) to generate synthetic handwritten digits in the style of the MNIST dataset. It includes both a model training component and a web application for interacting with the trained model.

## Project Overview

- **CGAN Model**: Generates realistic handwritten digits (0-9) based on a specified digit class
- **Web Application**: Allows users to select a digit and generate multiple synthetic images
- **Technology Stack**: PyTorch, Streamlit, Python

## Features

- Train a CGAN model on the MNIST dataset
- Generate multiple variations of any digit (0-9)
- Interactive web interface for digit generation
- Multi-GPU support for training

## Project Structure

- `cgan_model.py`: CGAN model architecture and training code
- `app.py`: Streamlit web application
- `requirements.txt`: Required Python packages
- `models/`: Directory containing the trained model weights
- `.gitignore`: Files to exclude from version control

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

## Training the CGAN Model

1. To train the model, uncomment the training line in the main function of `cgan_model.py`:

```python
generator, discriminator = train_cgan(epochs=epochs)
```

2. Run the training:

```bash
python cgan_model.py
```

3. The model will be saved in the `models/` directory as `generator_epoch_{epoch}.pth`.
4. Training progress is displayed with loss values and sample images.

## Running the Web Application Locally

1. Make sure the trained model exists in the `models/` directory.
2. Check that the model path in `app.py` matches your trained model:

```python
model_path = os.path.join(script_dir, "models", "generator_epoch_80.pth")  # Update this if needed
```

3. Run the Streamlit app:

```bash
streamlit run app.py
```

4. The web application will open in your browser, allowing you to:
   - Select a digit (0-9) to generate
   - Generate 5 unique images of the selected digit
   - View the generated images with captions

## Deploying to Streamlit Cloud

1. Create a GitHub repository and push your code:

```bash
git init
git add app.py cgan_model.py requirements.txt models/generator_epoch_80.pth .gitignore README.md
git commit -m "Initial commit of MNIST digit generator app"
git remote add origin https://github.com/your-username/mnist-digit-generator.git
git push -u origin main
```

2. Sign up at [https://streamlit.io/cloud](https://streamlit.io/cloud) with your GitHub account

3. From the Streamlit Cloud dashboard:
   - Click "New app"
   - Select your GitHub repository, branch, and main file (`app.py`)
   - Click "Deploy"

4. Your app will be available at a public URL provided by Streamlit

## Model Architecture

The CGAN consists of:

- **Generator**: Takes random noise + digit class as input and generates 28×28 grayscale images
- **Discriminator**: Evaluates if an image is real or fake, conditioned on the digit class
- **Training**: Adversarial loss with Adam optimizer

## Acknowledgments

- MNIST dataset from TorchVision
- GAN implementation inspired by PyTorch examples

## License

This project is available for educational and research purposes.
