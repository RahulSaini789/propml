from huggingface_hub import HfApi

# Delivery truck ka engine start karte hain
api = HfApi()

# 🔑 STEP 1: Apna VIP Pass (Token) yahan daalein
# Apne Hugging Face settings se 'propml-gurgaon' wala token copy karke yahan paste karein
api.token = "hf_YOUR_TOKEN_HERE"   

# 👤 STEP 2: Aapki HF Details (Added to file!)
YOUR_USERNAME = "Dumdigi"          
REPO_NAME = "propml-gurgaon"       

print(f"🚀 Uploading files to https://huggingface.co/{YOUR_USERNAME}/{REPO_NAME}...")

# 🧠 STEP 3: Upload the AI Brain (Model)
try:
    api.upload_file(
        path_or_fileobj = "models/current/model.pkl",
        path_in_repo    = "model.pkl",
        repo_id         = f"{YOUR_USERNAME}/{REPO_NAME}",
        repo_type       = "model",
    )
    print("✅ model.pkl uploaded successfully!")
except Exception as e:
    print(f"❌ Error uploading model: {e}")

# 📖 STEP 4: Upload the Memory (Feature Metadata)
try:
    api.upload_file(
        path_or_fileobj = "data/features/feature_metadata.json",
        path_in_repo    = "feature_metadata.json",
        repo_id         = f"{YOUR_USERNAME}/{REPO_NAME}",
        repo_type       = "model",
    )
    print("✅ feature_metadata.json uploaded successfully!")
except Exception as e:
    print(f"❌ Error uploading metadata: {e}")

print("\n🎉 Upload complete! Mission Successful!")
print(f"🌍 View your live model here: https://huggingface.co/{YOUR_USERNAME}/{REPO_NAME}")