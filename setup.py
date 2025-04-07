import subprocess
import sys
import os

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
    
    # Import after installation
    try:
        from sentence_transformers import SentenceTransformer
        
        # Download and cache the model
        print("Downloading sentence transformer model...")
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', cache_folder="./model_cache")
        print("Model downloaded and cached successfully!")
        return True
    except Exception as e:
        print(f"Error downloading model: {str(e)}")
        return False

def main():
    """Main setup function."""
    install_requirements()
    download_models()
    print("Setup completed successfully! You can now run: streamlit run main.py")

if __name__ == "__main__":
    main() 