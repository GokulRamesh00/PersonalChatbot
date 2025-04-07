import subprocess
import sys
import os
import time

def install_requirements():
    """Install required packages."""
    print("Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    print("All required packages installed successfully!")

def download_models():
    """Pre-download and cache models."""
    print("Downloading and caching models...")
    
    # Create cache directory
    os.makedirs("./model_cache", exist_ok=True)
    os.makedirs("./vector_store", exist_ok=True)
    
    # Set environment variables
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable GPU
    os.environ["USE_TORCH"] = "0"  # Disable torch for certain operations
    
    # Try multiple model options
    models_to_try = [
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/paraphrase-MiniLM-L3-v2"
    ]
    
    success = False
    for model_name in models_to_try:
        try:
            # Import after installation
            from sentence_transformers import SentenceTransformer
            
            # Download and cache the model
            print(f"Downloading sentence transformer model: {model_name}...")
            model = SentenceTransformer(model_name, cache_folder="./model_cache")
            
            # Create a simple test embedding to verify it works
            test_embedding = model.encode("This is a test sentence", convert_to_numpy=True)
            print(f"Model test successful with shape: {test_embedding.shape}")
            
            print(f"Model {model_name} downloaded and cached successfully!")
            success = True
            break
        except Exception as e:
            print(f"Error downloading model {model_name}: {str(e)}")
            print("Waiting 5 seconds before trying another model...")
            time.sleep(5)
    
    return success

def main():
    """Main setup function."""
    install_requirements()
    success = download_models()
    
    if success:
        print("Setup completed successfully! You can now run: streamlit run main.py")
    else:
        print("WARNING: Setup completed with issues. The application may still work but might fall back to alternative embeddings.")

if __name__ == "__main__":
    main() 