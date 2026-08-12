import os

def check_project_files():
    """Check if all required files are present"""
    
    required_files = [
        'app.py',
        'dataset_utils.py', 
        'requirements.txt',
        'templates/index.html',
        'models/psasinet_final.h5',
        'models/training_accuracy.txt'
    ]
    
    print("🔍 Checking PSASINet Project Files...")
    print("="*50)
    
    missing_files = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            if file_path.endswith('.h5'):
                size = os.path.getsize(file_path) / (1024*1024)  # MB
                print(f"✅ {file_path} - {size:.1f} MB")
            else:
                print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING!")
            missing_files.append(file_path)
    
    print("="*50)
    
    if missing_files:
        print(f"❌ {len(missing_files)} files missing!")
        print("Missing files:", missing_files)
        return False
    else:
        print("✅ All required files present!")
        return True

if __name__ == "__main__":
    check_project_files()