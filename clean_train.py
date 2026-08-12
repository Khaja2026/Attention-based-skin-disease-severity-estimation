import os
import shutil
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

def create_clean_dataset():
    """Create a completely clean dataset by copying only good files"""
    
    # Create clean dataset directory
    clean_path = 'data/clean_dataset'
    if os.path.exists(clean_path):
        shutil.rmtree(clean_path)
    
    os.makedirs(f'{clean_path}/train', exist_ok=True)
    os.makedirs(f'{clean_path}/val', exist_ok=True)
    
    # Copy only files with reasonable names
    original_path = 'data'
    
    for split in ['train', 'val']:
        split_path = os.path.join(original_path, split)
        if not os.path.exists(split_path):
            continue
            
        for class_dir in os.listdir(split_path):
            class_path = os.path.join(split_path, class_dir)
            if not os.path.isdir(class_path):
                continue
                
            # Create clean class directory
            clean_class_path = os.path.join(clean_path, split, class_dir)
            os.makedirs(clean_class_path, exist_ok=True)
            
            copied_count = 0
            for filename in os.listdir(class_path):
                # Skip problematic files
                if (len(filename) > 100 or 
                    'natural-treatment' in filename or
                    not filename.lower().endswith(('.jpg', '.jpeg', '.png'))):
                    continue
                
                src_file = os.path.join(class_path, filename)
                dst_file = os.path.join(clean_class_path, filename)
                
                try:
                    shutil.copy2(src_file, dst_file)
                    copied_count += 1
                except:
                    continue
            
            print(f"Copied {copied_count} files to {split}/{class_dir}")
    
    print("✅ Clean dataset created!")
    return clean_path

def simple_train():
    """Simple training with clean dataset"""
    
    # Create clean dataset
    clean_path = create_clean_dataset()
    
    # Try to load existing model, create new if fails
    try:
        model = tf.keras.models.load_model('models/psasinet_compatible.h5')
        print("✅ Loaded compatible model")
    except:
        try:
            model = tf.keras.models.load_model('models/psasinet_final (1).h5')
            print("✅ Loaded original model")
        except:
            print("Creating new model...")
            # Create new model
            base = tf.keras.applications.EfficientNetB3(weights='imagenet', include_top=False, input_shape=(224,224,3))
            x = base.output
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            x = tf.keras.layers.Dense(512, activation='relu')(x)
            x = tf.keras.layers.Dropout(0.5)(x)
            outputs = tf.keras.layers.Dense(9, activation='softmax')(x)
            model = tf.keras.Model(inputs=base.input, outputs=outputs)
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Enhanced data generators with augmentation
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )
    
    train_gen = datagen.flow_from_directory(
        f'{clean_path}/train',
        target_size=(224, 224),
        batch_size=8,
        class_mode='categorical',
        subset='training'
    )
    
    val_gen = datagen.flow_from_directory(
        f'{clean_path}/train',
        target_size=(224, 224),
        batch_size=8,
        class_mode='categorical',
        subset='validation'
    )
    
    # Train for 50 more epochs with enhanced callbacks
    callbacks = [
        ReduceLROnPlateau(monitor='val_accuracy', factor=0.3, patience=3, min_lr=1e-8, verbose=1),
        EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1)
    ]
    
    print("🚀 Starting 10 epoch enhanced training...")
    history = model.fit(
        train_gen,
        epochs=10,  # Training for 10 epochs
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save model and accuracy
    model.save('models/psasinet_compatible.h5')
    model.save('models/psasinet_final.h5')
    
    best_accuracy = max(history.history['val_accuracy'])
    with open('models/training_accuracy.txt', 'w') as f:
        f.write(f"{best_accuracy*100:.1f}%")
    
    print(f"✅ Training complete! Accuracy: {best_accuracy*100:.1f}%")

if __name__ == "__main__":
    simple_train()