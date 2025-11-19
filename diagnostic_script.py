"""
Diagnostic Script - Test Your Gastric Cancer Model
Run this to verify everything is working correctly
"""

import numpy as np
import tensorflow as tf
from PIL import Image
import os
import sys

print("=" * 70)
print("🔬 GASTRIC CANCER MODEL DIAGNOSTIC TEST")
print("=" * 70)

# ============================================================================
# 1. CHECK MODEL FILE
# ============================================================================
print("\n📦 Step 1: Checking Model File...")

MODEL_PATH = 'gastric_working_final.keras'

if not os.path.exists(MODEL_PATH):
    print(f"❌ ERROR: Model file not found: {MODEL_PATH}")
    print("   Please ensure the model file is in the current directory.")
    sys.exit(1)

file_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)
print(f"✅ Model file found: {file_size:.2f} MB")

if file_size < 1:
    print("⚠️  WARNING: Model file seems too small. Expected ~5-6 MB")
elif file_size > 20:
    print("⚠️  WARNING: Model file seems too large. Expected ~5-6 MB")
else:
    print("✅ Model file size looks correct")

# ============================================================================
# 2. LOAD MODEL
# ============================================================================
print("\n🧠 Step 2: Loading Model...")

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ ERROR loading model: {e}")
    sys.exit(1)

# Check model details
print(f"   Input shape: {model.input_shape}")
print(f"   Output shape: {model.output_shape}")
print(f"   Parameters: {model.count_params():,}")

if model.input_shape != (None, 120, 120, 3):
    print("⚠️  WARNING: Input shape doesn't match expected (None, 120, 120, 3)")

if model.output_shape != (None, 1):
    print("⚠️  WARNING: Output shape doesn't match expected (None, 1)")

# ============================================================================
# 3. TEST WITH DUMMY DATA
# ============================================================================
print("\n🧪 Step 3: Testing with Dummy Data...")

# Create dummy images
print("   Creating synthetic test images...")

# Dummy "normal" image (lighter, more uniform)
normal_dummy = np.random.randint(180, 250, (120, 120, 3), dtype=np.uint8)

# Dummy "abnormal" image (darker, more varied)
abnormal_dummy = np.random.randint(50, 150, (120, 120, 3), dtype=np.uint8)

# Preprocess like training
normal_array = normal_dummy.astype(np.float32) / 255.0
normal_array = np.expand_dims(normal_array, axis=0)

abnormal_array = abnormal_dummy.astype(np.float32) / 255.0
abnormal_array = np.expand_dims(abnormal_array, axis=0)

# Predict
try:
    pred_normal = model.predict(normal_array, verbose=0)[0][0]
    pred_abnormal = model.predict(abnormal_array, verbose=0)[0][0]
    
    print(f"   Dummy Normal prediction: {pred_normal:.4f}")
    print(f"   Dummy Abnormal prediction: {pred_abnormal:.4f}")
    print("✅ Model can make predictions")
except Exception as e:
    print(f"❌ ERROR during prediction: {e}")
    sys.exit(1)

# ============================================================================
# 4. TEST WITH ACTUAL IMAGE (if provided)
# ============================================================================
print("\n📸 Step 4: Testing with Actual Images...")

test_images = []

# Look for test images
for ext in ['.jpg', '.jpeg', '.png']:
    test_images.extend([f for f in os.listdir('.') if f.lower().endswith(ext)])

if len(test_images) == 0:
    print("⚠️  No test images found in current directory")
    print("   Place a test image (normal.jpg or abnormal.jpg) in the same directory")
else:
    print(f"   Found {len(test_images)} image(s) to test")
    
    for img_path in test_images[:5]:  # Test first 5
        try:
            print(f"\n   Testing: {img_path}")
            
            # Load image
            img = Image.open(img_path).convert('RGB')
            original_size = img.size
            print(f"      Original size: {original_size}")
            
            # Preprocess (matching training)
            img = img.resize((120, 120), Image.Resampling.LANCZOS)
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Predict
            pred = model.predict(img_array, verbose=0)[0][0]
            
            # Interpret with different thresholds
            print(f"      Raw score: {pred:.4f}")
            print(f"      With threshold 0.4: {'Abnormal' if pred > 0.4 else 'Normal'}")
            print(f"      With threshold 0.5: {'Abnormal' if pred > 0.5 else 'Normal'}")
            
            if pred < 0.3:
                print(f"      ✅ Strong Normal prediction")
            elif pred < 0.4:
                print(f"      ⚠️  Weak Normal prediction (close to threshold)")
            elif pred < 0.6:
                print(f"      ⚠️  Weak Abnormal prediction (close to threshold)")
            else:
                print(f"      ✅ Strong Abnormal prediction")
                
        except Exception as e:
            print(f"      ❌ ERROR: {e}")

# ============================================================================
# 5. TEST PREPROCESSING METHODS
# ============================================================================
print("\n🔄 Step 5: Comparing Preprocessing Methods...")

if len(test_images) > 0:
    test_img = test_images[0]
    
    try:
        # Method 1: PIL (current inference)
        img_pil = Image.open(test_img).convert('RGB')
        img_pil = img_pil.resize((120, 120), Image.Resampling.LANCZOS)
        arr_pil = np.array(img_pil, dtype=np.float32) / 255.0
        
        # Method 2: Keras (training)
        from tensorflow.keras.preprocessing import image as keras_image
        img_keras = keras_image.load_img(test_img, target_size=(120, 120))
        arr_keras = keras_image.img_to_array(img_keras) / 255.0
        
        # Compare
        diff_mean = np.abs(arr_pil - arr_keras).mean()
        diff_max = np.abs(arr_pil - arr_keras).max()
        
        print(f"   Testing: {test_img}")
        print(f"   Mean difference: {diff_mean:.6f}")
        print(f"   Max difference: {diff_max:.6f}")
        
        if diff_mean < 0.01:
            print("   ✅ Preprocessing methods are virtually identical")
        elif diff_mean < 0.05:
            print("   ⚠️  Small difference - should be OK")
        else:
            print("   ❌ Significant difference - preprocessing mismatch!")
            
    except Exception as e:
        print(f"   ⚠️  Could not compare: {e}")

# ============================================================================
# 6. CONFIGURATION CHECK
# ============================================================================
print("\n⚙️  Step 6: Configuration Recommendations...")

print("\n   app.py Configuration:")
print("   " + "-" * 60)
print("   IMG_SIZE = (120, 120)                    # ✅ Correct")
print("   THRESHOLD = 0.4                          # ✅ From training")
print("   APPLY_STAIN_NORMALIZATION = False        # ✅ No normalization")
print("   HISTOLOGY_CHECK = True                   # Can disable for testing")
print("   " + "-" * 60)

print("\n   If normal images are misclassified:")
print("   1. Set HISTOLOGY_CHECK = False (disable validation)")
print("   2. Try THRESHOLD = 0.5 (more conservative)")
print("   3. Verify test images are similar to training images")

# ============================================================================
# 7. SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("📊 DIAGNOSTIC SUMMARY")
print("=" * 70)

checks = {
    "Model file exists": os.path.exists(MODEL_PATH),
    "Model loads correctly": True,
    "Model can predict": True,
    "Input shape correct": model.input_shape == (None, 120, 120, 3),
    "Output shape correct": model.output_shape == (None, 1),
}

all_passed = all(checks.values())

for check, passed in checks.items():
    status = "✅" if passed else "❌"
    print(f"{status} {check}")

if all_passed:
    print("\n✅ All checks passed! Model is ready to use.")
    print("\nNext steps:")
    print("1. Test with actual histology images from your training set")
    print("2. If issues persist, adjust THRESHOLD (try 0.5)")
    print("3. Check if test images are from the same distribution as training")
else:
    print("\n⚠️  Some checks failed. Please review the issues above.")

print("\n" + "=" * 70)
print("🎯 To test in web interface:")
print("   1. python app.py")
print("   2. Open http://localhost:5000")
print("   3. Upload a known Normal image from your training set")
print("   4. Check if prediction matches expectation")
print("=" * 70)

# ============================================================================
# 8. SAVE TEST REPORT
# ============================================================================
print("\n💾 Saving diagnostic report...")

report = f"""
GASTRIC CANCER MODEL DIAGNOSTIC REPORT
======================================
Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

MODEL INFORMATION
-----------------
File: {MODEL_PATH}
Size: {file_size:.2f} MB
Input Shape: {model.input_shape}
Output Shape: {model.output_shape}
Parameters: {model.count_params():,}

CHECKS
------
"""

for check, passed in checks.items():
    report += f"{'✅' if passed else '❌'} {check}\n"

report += f"""

RECOMMENDATIONS
---------------
1. Configuration (app.py):
   - IMG_SIZE = (120, 120)
   - THRESHOLD = 0.4 (or try 0.5)
   - APPLY_STAIN_NORMALIZATION = False
   - HISTOLOGY_CHECK = True (or False to disable)

2. Testing:
   - Use images from your training dataset first
   - Verify preprocessing matches training
   - Check prediction scores (should be clear, not borderline)

3. If normal images misclassify:
   - Disable validation: HISTOLOGY_CHECK = False
   - Increase threshold: THRESHOLD = 0.5
   - Test with training images to verify model works

STATUS: {'All systems operational' if all_passed else 'Issues detected'}
"""

with open('diagnostic_report.txt', 'w') as f:
    f.write(report)

print("✅ Report saved: diagnostic_report.txt")
print("\n✨ Diagnostic complete!")