import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
import os

def create_data_generators():
    """Create data generators for continued training"""
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest',
        validation_split=0.2
    )
    
    # Use validation folder for training (clean data)
    train_generator = train_datagen.flow_from_directory(
        'data/archive (1)/Split_smol/val',
        target_size=(224, 224),
        batch_size=8,
        class_mode='categorical',
        subset='training'
    )
    
    val_generator = train_datagen.flow_from_directory(
        'data/archive (1)/Split_smol/val',
        target_size=(224, 224),
        batch_size=8,
        class_mode='categorical',
        subset='validation'
    )
    
    return train_generator, val_generator

def continue_training_50_epochs():
    """Continue training existing model for 50 more epochs"""
    
    print("🔄 Loading existing PSASINet model...")
    
    # Load the existing model
    try:
        model = tf.keras.models.load_model('models/psasinet_final.h5')
        print("✅ Model loaded successfully!")
    except:
        print("❌ Could not load existing model. Please train the base model first.")
        return None
    
    # Lower learning rate for fine-tuning
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=0.0001,  # Lower LR for continued training
        weight_decay=0.0001
    )
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks for continued training
    callbacks = [
        ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.3,
            patience=3,
            min_lr=1e-8,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            'models/psasinet_continued.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Load data generators
    print("📊 Loading data generators...")
    train_gen, val_gen = create_data_generators()
    
    print("🚀 Starting continued training for 50 epochs...")
    
    # Continue training for 50 epochs
    history = model.fit(
        train_gen,
        epochs=50,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save the improved model
    model.save('models/psasinet_final.h5')  # Overwrite with improved version
    
    # Update accuracy file
    best_accuracy = max(history.history['val_accuracy'])
    accuracy_percentage = f"{best_accuracy*100:.1f}%"
    
    with open('models/training_accuracy.txt', 'w') as f:
        f.write(accuracy_percentage)
    
    print(f"🎯 CONTINUED TRAINING COMPLETE!")
    print(f"📈 Best Accuracy: {accuracy_percentage}")
    print(f"💾 Model saved as: models/psasinet_final.h5")
    
    return model, history

if __name__ == "__main__":
    print("🔄 PSASINet - Continue Training for 50 Epochs")
    print("=" * 50)
    
    model, history = continue_training_50_epochs()
    
    if model:
        print("\n✅ Training completed successfully!")
        print("🔍 Check models/training_accuracy.txt for final accuracy")
    else:
        print("\n❌ Training failed. Please check your setup.")