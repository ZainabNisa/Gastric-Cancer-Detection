"""
Test script to debug Grad-CAM with your gastric cancer model
Run this to see exactly where Grad-CAM fails

Usage: python test_gradcam.py
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF logging

import numpy as np
import tensorflow as tf

print("=" * 70)
print("GRAD-CAM DEBUG SCRIPT")
print("=" * 70)

# Load model
print("\n1. Loading model...")
try:
    model = tf.keras.models.load_model('gastric_working_final.keras')
    print(f"   ✅ Model loaded: {model.input_shape} → {model.output_shape}")
except Exception as e:
    print(f"   ❌ Failed to load model: {e}")
    exit(1)

# Inspect architecture
print("\n2. Model Architecture:")
print("-" * 70)
for i, layer in enumerate(model.layers):
    print(f"   {i}: {layer.name} ({layer.__class__.__name__})")
    if hasattr(layer, 'layers'):
        for j, sublayer in enumerate(layer.layers):
            print(f"      {j}: {sublayer.name} ({sublayer.__class__.__name__})")
            if hasattr(sublayer, 'output_shape'):
                try:
                    print(f"         Output shape: {sublayer.output_shape}")
                except:
                    pass

# Find Conv layers
print("\n3. Finding Convolutional Layers:")
print("-" * 70)

def find_conv_layers(model, parent_name=""):
    conv_layers = []
    for layer in model.layers:
        full_name = f"{parent_name}/{layer.name}" if parent_name else layer.name
        if hasattr(layer, 'layers'):
            # Nested model
            conv_layers.extend(find_conv_layers(layer, full_name))
        elif 'Conv' in layer.__class__.__name__:
            conv_layers.append((full_name, layer))
    return conv_layers

conv_layers = find_conv_layers(model)

if not conv_layers:
    print("   ❌ NO CONVOLUTIONAL LAYERS FOUND!")
    print("   This model cannot use Grad-CAM.")
    exit(1)

print(f"   Found {len(conv_layers)} Conv layers:")
for name, layer in conv_layers:
    print(f"   ✓ {name} ({layer.__class__.__name__})")

target_layer_name = conv_layers[-1][0]
print(f"\n   🎯 Target layer: {target_layer_name}")

# Test forward pass
print("\n4. Testing Forward Pass:")
print("-" * 70)
try:
    dummy_input = np.random.rand(1, 120, 120, 3).astype(np.float32)
    print(f"   Input shape: {dummy_input.shape}")
    
    prediction = model.predict(dummy_input, verbose=0)
    print(f"   ✅ Prediction works: {prediction[0][0]:.6f}")
except Exception as e:
    print(f"   ❌ Forward pass failed: {e}")
    exit(1)

# Test Grad-CAM
print("\n5. Testing Grad-CAM:")
print("-" * 70)

try:
    img_tensor = tf.convert_to_tensor(dummy_input, dtype=tf.float32)
    conv_output = None
    
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(img_tensor)
        x = img_tensor
        
        print(f"   Starting manual forward pass...")
        
        # Parse layer path
        if '/' in target_layer_name:
            parts = target_layer_name.split('/')
            parent_name, target_name = parts[0], parts[1]
            print(f"   Nested layer: parent={parent_name}, target={target_name}")
            
            # Go through layers
            for i, layer in enumerate(model.layers):
                print(f"   Layer {i}: {layer.name}")
                
                if layer.name == parent_name:
                    print(f"      → Entering nested model...")
                    # Navigate nested layers
                    for j, sublayer in enumerate(layer.layers):
                        print(f"      Sublayer {j}: {sublayer.name}")
                        x = sublayer(x, training=False)
                        if sublayer.name == target_name:
                            conv_output = x
                            print(f"      ✅ Captured conv output: {conv_output.shape}")
                else:
                    x = layer(x, training=False)
        else:
            # Top-level layer
            for i, layer in enumerate(model.layers):
                print(f"   Layer {i}: {layer.name}")
                x = layer(x, training=False)
                if layer.name == target_layer_name:
                    conv_output = x
                    print(f"   ✅ Captured conv output: {conv_output.shape}")
        
        final_output = x
        print(f"   Final output shape: {final_output.shape}")
        
        # Loss
        loss = final_output[:, 0]
        print(f"   Loss: {loss.numpy()[0]:.6f}")
    
    if conv_output is None:
        print(f"   ❌ Failed to capture conv output from {target_layer_name}")
        exit(1)
    
    print(f"\n   Computing gradients...")
    grads = tape.gradient(loss, conv_output)
    
    if grads is None:
        print(f"   ❌ Gradients are None!")
        exit(1)
    
    print(f"   ✅ Gradients: {grads.shape}")
    print(f"      Range: [{grads.numpy().min():.6f}, {grads.numpy().max():.6f}]")
    
    # Compute heatmap
    print(f"\n   Computing heatmap...")
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    print(f"   Pooled grads: {pooled_grads.shape}")
    
    conv_output_np = conv_output[0].numpy()
    pooled_grads_np = pooled_grads.numpy()
    
    for i in range(len(pooled_grads_np)):
        conv_output_np[:, :, i] *= pooled_grads_np[i]
    
    heatmap = np.mean(conv_output_np, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    
    print(f"   ✅ Heatmap: {heatmap.shape}")
    print(f"      Range: [{heatmap.min():.3f}, {heatmap.max():.3f}]")
    print(f"      Non-zero values: {np.count_nonzero(heatmap)}/{heatmap.size}")
    
    print("\n" + "=" * 70)
    print("✅✅✅ GRAD-CAM WORKS! ✅✅✅")
    print("=" * 70)
    print("\nYou can now use this Grad-CAM implementation in your Flask app.")
    
except Exception as e:
    print(f"\n❌ Grad-CAM failed: {e}")
    print("\nFull error:")
    import traceback
    traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("DEBUGGING TIPS:")
    print("=" * 70)
    print("1. Check if the model has nested Sequential layers")
    print("2. Verify Conv layers can be accessed")
    print("3. Try updating TensorFlow: pip install --upgrade tensorflow")
    print("4. Share the full error output for more help")