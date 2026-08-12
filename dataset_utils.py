import os
import json

def get_dataset_info():
    """Dynamically get dataset information from actual files"""
    dataset_path = 'data'
    
    if not os.path.exists(dataset_path):
        # Return default values for demo without dataset
        return {
            'training_samples': 878,
            'validation_samples': 219,
            'total_samples': 1097,
            'classes': 9,
            'class_names': [
                'actinic keratosis',
                'atopic dermatitis', 
                'benign keratosis',
                'dermatofibroma',
                'melanocytic nevus',
                'melanoma',
                'squamous cell carcinoma',
                'tinea ringworm candidiasis',
                'vascular lesion'
            ]
        }
    
    # Count training samples
    train_path = os.path.join(dataset_path, 'train')
    val_path = os.path.join(dataset_path, 'val')
    
    train_count = 0
    val_count = 0
    class_names = []
    
    if os.path.exists(train_path):
        for class_dir in os.listdir(train_path):
            class_path = os.path.join(train_path, class_dir)
            if os.path.isdir(class_path):
                class_names.append(class_dir)
                train_count += len([f for f in os.listdir(class_path) 
                                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    if os.path.exists(val_path):
        for class_dir in os.listdir(val_path):
            class_path = os.path.join(val_path, class_dir)
            if os.path.isdir(class_path):
                val_count += len([f for f in os.listdir(class_path) 
                               if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    return {
        'training_samples': train_count,
        'validation_samples': val_count,
        'total_samples': train_count + val_count,
        'classes': len(class_names),
        'class_names': sorted(class_names)
    }

def get_disease_mapping():
    """Dynamically create disease mapping from dataset structure"""
    dataset_info = get_dataset_info()
    class_names = dataset_info['class_names']
    
    # Severity mapping based on medical knowledge
    severity_map = {
        'melanoma': 'Severe',
        'squamous cell carcinoma': 'Severe',
        'actinic keratosis': 'Moderate',
        'tinea ringworm candidiasis': 'Moderate',
        'atopic dermatitis': 'Mild',
        'vascular lesion': 'Mild',
        'benign keratosis': 'Benign',
        'dermatofibroma': 'Benign',
        'melanocytic nevus': 'Benign'
    }
    
    color_map = {
        'Severe': '#dc3545',
        'Moderate': '#fd7e14',
        'Mild': '#ffc107',
        'Benign': '#28a745'
    }
    
    disease_mapping = {}
    for i, class_name in enumerate(class_names):
        # Determine severity based on class name
        severity = 'Unknown'
        for key, sev in severity_map.items():
            if key.lower() in class_name.lower():
                severity = sev
                break
        
        disease_mapping[i] = {
            'name': class_name,
            'severity': severity,
            'color': color_map.get(severity, '#6c757d')
        }
    
    return disease_mapping

if __name__ == "__main__":
    # Test the functions
    info = get_dataset_info()
    mapping = get_disease_mapping()
    
    print("Dataset Info:", info)
    print("Disease Mapping:", mapping)