import sys
import os
import importlib

def check_dependencies():
    required = [
        "fitz", "pytesseract", "PIL", "sentence_transformers", "qdrant_client",
        "openai", "langchain", "fastapi", "uvicorn"
    ]
    missing = []
    for pkg in required:
        try:
            importlib.import_module(pkg)
        except ImportError:
            # Handle package name mapping if necessary
            if pkg == "fitz":
                try:
                    import fitz
                except ImportError:
                    missing.append("pymupdf")
            elif pkg == "PIL":
                try:
                    import PIL
                except ImportError:
                    missing.append("Pillow")
            else:
                missing.append(pkg)
    
    if missing:
        print(f"❌ Missing dependencies: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False
    print("✅ Python dependencies installed.")
    return True

def check_env():
    if not os.path.exists(".env"):
        print("❌ .env file missing. Copy .env.example to .env")
        return False
    
    with open(".env") as f:
        content = f.read()
        if "your_openai_api_key" in content:
            print("⚠️ .env file contains default values. Please update OPENAI_API_KEY.")
        else:
            print("✅ .env file present.")
    return True

if __name__ == "__main__":
    print("Verifying PDF CLA-RA setup...")
    deps_ok = check_dependencies()
    env_ok = check_env()
    
    if deps_ok and env_ok:
        print("\n🚀 Verification complete! You can start the app with:")
        print("docker-compose up -d")
        print("uvicorn app.main:app --reload")
    else:
        print("\n❌ Setup incomplete. Please fix errors above.")
