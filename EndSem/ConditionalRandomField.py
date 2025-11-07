import numpy as np
import cv2
import os
import gc
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax
import matplotlib.pyplot as plt
from glob import glob


# ============================================================================
# MEMORY OPTIMIZATION FUNCTIONS
# ============================================================================

def resize_image(image, max_size=512):
    """Resize regular image (fundus) to reduce memory usage"""
    height, width = image.shape[:2]
    if max(height, width) > max_size:
        scale = max_size / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return image


def resize_mask(mask, max_size=512):
    """Resize mask using nearest neighbor to preserve discrete labels"""
    height, width = mask.shape[:2]
    if max(height, width) > max_size:
        scale = max_size / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        resized = cv2.resize(mask, (new_width, new_height), 
                           interpolation=cv2.INTER_NEAREST)
        return resized
    return mask


def remap_mask_labels(mask):
    """Remap mask values to exactly 0, 128, or 255"""
    output = np.zeros_like(mask)
    output[mask < 64] = 0
    output[(mask >= 64) & (mask < 192)] = 128
    output[mask >= 192] = 255
    return output


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def extract_features_with_coords(image, normalize_coords=True, use_both_colorspaces=True):
    """Extract color and spatial features from fundus image"""
    height, width = image.shape[:2]
    
    # Extract RGB features
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb_features = image_rgb.reshape(-1, 3)
    
    if use_both_colorspaces:
        # Extract HSV features
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv_features = image_hsv.reshape(-1, 3)
        color_features = np.hstack([rgb_features, hsv_features])
    else:
        color_features = rgb_features
    
    # Create coordinate grids
    if normalize_coords:
        y_coords, x_coords = np.meshgrid(
            np.linspace(0, 1, height),
            np.linspace(0, 1, width),
            indexing='ij'
        )
    else:
        y_coords, x_coords = np.meshgrid(
            range(height), range(width), indexing='ij'
        )
    
    x_features = x_coords.flatten().reshape(-1, 1)
    y_features = y_coords.flatten().reshape(-1, 1)
    
    # Combine all features
    all_features = np.hstack([color_features, x_features, y_features])
    
    return all_features


# ============================================================================
# MEMORY-EFFICIENT TRAINING
# ============================================================================

def train_naive_bayes_memory_efficient(fundus_files, mask_files, 
                                       use_both_colorspaces=True, 
                                       batch_size=5, max_size=512):
    """Memory-efficient training with batching"""
    print("Training with memory-efficient batching...")
    scaler = StandardScaler()
    clf = GaussianNB()
    
    n_batches = (len(fundus_files) + batch_size - 1) // batch_size
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(fundus_files))
        
        print(f"Processing batch {batch_idx + 1}/{n_batches} (images {start_idx+1}-{end_idx})...")
        
        X_batch = []
        y_batch = []
        
        for i in range(start_idx, end_idx):
            fundus = cv2.imread(fundus_files[i])
            mask = cv2.imread(mask_files[i], cv2.IMREAD_GRAYSCALE)
            
            if fundus is None or mask is None:
                print(f"  Warning: Could not load image pair {i+1}")
                continue
            
            # Resize
            fundus = resize_image(fundus, max_size=max_size)
            mask = resize_mask(mask, max_size=max_size)
            mask = remap_mask_labels(mask)
            
            # Extract features
            features = extract_features_with_coords(
                fundus, 
                normalize_coords=True,
                use_both_colorspaces=use_both_colorspaces
            )
            labels = mask.flatten()
            
            X_batch.append(features)
            y_batch.append(labels)
            
            del fundus, mask
        
        if len(X_batch) == 0:
            continue
            
        X_batch = np.vstack(X_batch)
        y_batch = np.concatenate(y_batch)
        
        print(f"  Batch samples: {X_batch.shape[0]}")
        
        # Partial fit
        if batch_idx == 0:
            X_batch_scaled = scaler.fit_transform(X_batch)
            clf.partial_fit(X_batch_scaled, y_batch, classes=np.array([0, 128, 255]))
        else:
            X_batch_scaled = scaler.transform(X_batch)
            clf.partial_fit(X_batch_scaled, y_batch)
        
        del X_batch, y_batch, X_batch_scaled
        gc.collect()
    
    print("Training complete!")
    return clf, scaler


# ============================================================================
# CRF FUNCTIONS
# ============================================================================

def get_probability_map(clf, scaler, image, use_both_colorspaces=True):
    """Get probability map from Naive Bayes classifier"""
    height, width = image.shape[:2]
    
    features = extract_features_with_coords(
        image,
        normalize_coords=True,
        use_both_colorspaces=use_both_colorspaces
    )
    
    features_scaled = scaler.transform(features)
    prob = clf.predict_proba(features_scaled)
    
    n_classes = prob.shape[1]
    prob = prob.T.reshape(n_classes, height, width)
    
    return prob


def apply_dense_crf(image, probabilities, n_iters=5, 
                    sxy_gaussian=3, compat_gaussian=3,
                    sxy_bilateral=80, srgb_bilateral=13, compat_bilateral=10):
    """Apply Dense CRF refinement"""
    height, width = image.shape[:2]
    n_classes = probabilities.shape[0]
    
    d = dcrf.DenseCRF2D(width, height, n_classes)
    
    unary = unary_from_softmax(probabilities)
    unary = np.ascontiguousarray(unary)
    d.setUnaryEnergy(unary)
    
    d.addPairwiseGaussian(sxy=sxy_gaussian, compat=compat_gaussian,
                          kernel=dcrf.DIAG_KERNEL,
                          normalization=dcrf.NORMALIZE_SYMMETRIC)
    
    d.addPairwiseBilateral(sxy=sxy_bilateral, srgb=srgb_bilateral,
                          rgbim=image.astype(np.uint8),
                          compat=compat_bilateral,
                          kernel=dcrf.DIAG_KERNEL,
                          normalization=dcrf.NORMALIZE_SYMMETRIC)
    
    Q = d.inference(n_iters)
    map_result = np.argmax(Q, axis=0).reshape((height, width))
    
    return map_result


def map_labels_back(crf_output, label_mapping):
    """Map CRF integer labels back to original mask values"""
    final_mask = np.zeros_like(crf_output, dtype=np.uint8)
    for int_label, orig_value in label_mapping.items():
        final_mask[crf_output == int_label] = orig_value
    return final_mask


def predict_with_crf(clf, scaler, test_image, use_both_colorspaces=True,
                     apply_crf=True, crf_params=None):
    """Predict segmentation with optional CRF refinement"""
    probabilities = get_probability_map(clf, scaler, test_image, use_both_colorspaces)
    
    class_labels = clf.classes_
    int_to_label = {i: label for i, label in enumerate(class_labels)}
    
    naive_prediction = np.argmax(probabilities, axis=0)
    naive_bayes_mask = map_labels_back(naive_prediction, int_to_label)
    
    if not apply_crf:
        return naive_bayes_mask, naive_bayes_mask
    
    print("  Applying Dense CRF refinement...")
    
    image_rgb = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    
    if crf_params is None:
        crf_params = {
            'n_iters': 5,
            'sxy_gaussian': 3,
            'compat_gaussian': 3,
            'sxy_bilateral': 80,
            'srgb_bilateral': 13,
            'compat_bilateral': 10
        }
    
    crf_output = apply_dense_crf(image_rgb, probabilities, **crf_params)
    final_mask = map_labels_back(crf_output, int_to_label)
    
    return final_mask, naive_bayes_mask


# ============================================================================
# EVALUATION
# ============================================================================

def calculate_dice_coefficient(pred, gt, label):
    """Calculate Dice coefficient for a specific class"""
    pred_label = (pred == label).astype(float)
    gt_label = (gt == label).astype(float)
    
    intersection = np.sum(pred_label * gt_label)
    dice = (2.0 * intersection) / (np.sum(pred_label) + np.sum(gt_label) + 1e-8)
    
    return dice


def evaluate_segmentation(pred_mask, gt_mask):
    """Evaluate segmentation performance"""
    accuracy = accuracy_score(gt_mask.flatten(), pred_mask.flatten())
    
    dice_disc = calculate_dice_coefficient(pred_mask, gt_mask, 0)
    dice_cup = calculate_dice_coefficient(pred_mask, gt_mask, 128)
    dice_bg = calculate_dice_coefficient(pred_mask, gt_mask, 255)
    
    return {
        'accuracy': accuracy,
        'dice_disc': dice_disc,
        'dice_cup': dice_cup,
        'dice_background': dice_bg,
        'dice_mean': (dice_disc + dice_cup + dice_bg) / 3
    }


# ============================================================================
# METRICS VISUALIZATION
# ============================================================================

def save_comparison_metrics_as_image(metrics_nb, metrics_crf, image_name, save_path):
    """Save comparison of Naive Bayes and CRF metrics as an image"""
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axis('tight')
    ax.axis('off')
    
    improvements = {
        key: metrics_crf[key] - metrics_nb[key] 
        for key in metrics_nb.keys()
    }
    
    metrics_data = [
        ['Metric', 'Naive Bayes', 'With CRF', 'Improvement'],
        ['Accuracy', 
         f"{metrics_nb['accuracy']:.4f}",
         f"{metrics_crf['accuracy']:.4f}",
         f"{improvements['accuracy']:+.4f}"],
        ['Dice - Disc', 
         f"{metrics_nb['dice_disc']:.4f}",
         f"{metrics_crf['dice_disc']:.4f}",
         f"{improvements['dice_disc']:+.4f}"],
        ['Dice - Cup', 
         f"{metrics_nb['dice_cup']:.4f}",
         f"{metrics_crf['dice_cup']:.4f}",
         f"{improvements['dice_cup']:+.4f}"],
        ['Dice - Background', 
         f"{metrics_nb['dice_background']:.4f}",
         f"{metrics_crf['dice_background']:.4f}",
         f"{improvements['dice_background']:+.4f}"],
        ['Mean Dice', 
         f"{metrics_nb['dice_mean']:.4f}",
         f"{metrics_crf['dice_mean']:.4f}",
         f"{improvements['dice_mean']:+.4f}"]
    ]
    
    table = ax.table(cellText=metrics_data,
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.35, 0.25, 0.25, 0.25])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Header styling
    for i in range(4):
        table[(0, i)].set_facecolor('#2196F3')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color improvement column
    for i in range(1, len(metrics_data)):
        improvement_val = float(metrics_data[i][3])
        if improvement_val > 0:
            table[(i, 3)].set_facecolor('#c8e6c9')
        elif improvement_val < 0:
            table[(i, 3)].set_facecolor('#ffcdd2')
        
        if i % 2 == 0:
            for j in range(3):
                table[(i, j)].set_facecolor('#f5f5f5')
    
    plt.title(f'Metrics Comparison - {image_name}',
              fontsize=16, fontweight='bold', pad=20)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def save_summary_metrics_as_image(metrics_nb_list, metrics_crf_list, save_path):
    """Save overall summary metrics as a table image"""
    if not metrics_crf_list:
        return
    
    # Calculate averages
    avg_nb = {
        'accuracy': np.mean([m['accuracy'] for m in metrics_nb_list]),
        'dice_disc': np.mean([m['dice_disc'] for m in metrics_nb_list]),
        'dice_cup': np.mean([m['dice_cup'] for m in metrics_nb_list]),
        'dice_background': np.mean([m['dice_background'] for m in metrics_nb_list]),
        'dice_mean': np.mean([m['dice_mean'] for m in metrics_nb_list])
    }
    
    avg_crf = {
        'accuracy': np.mean([m['accuracy'] for m in metrics_crf_list]),
        'dice_disc': np.mean([m['dice_disc'] for m in metrics_crf_list]),
        'dice_cup': np.mean([m['dice_cup'] for m in metrics_crf_list]),
        'dice_background': np.mean([m['dice_background'] for m in metrics_crf_list]),
        'dice_mean': np.mean([m['dice_mean'] for m in metrics_crf_list])
    }
    
    improvements = {key: avg_crf[key] - avg_nb[key] for key in avg_nb.keys()}
    
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axis('tight')
    ax.axis('off')
    
    metrics_data = [
        ['Metric', 'Naive Bayes', 'With CRF', 'Improvement'],
        ['Accuracy', 
         f"{avg_nb['accuracy']:.4f}",
         f"{avg_crf['accuracy']:.4f}",
         f"{improvements['accuracy']:+.4f}"],
        ['Dice - Disc', 
         f"{avg_nb['dice_disc']:.4f}",
         f"{avg_crf['dice_disc']:.4f}",
         f"{improvements['dice_disc']:+.4f}"],
        ['Dice - Cup', 
         f"{avg_nb['dice_cup']:.4f}",
         f"{avg_crf['dice_cup']:.4f}",
         f"{improvements['dice_cup']:+.4f}"],
        ['Dice - Background', 
         f"{avg_nb['dice_background']:.4f}",
         f"{avg_crf['dice_background']:.4f}",
         f"{improvements['dice_background']:+.4f}"],
        ['Mean Dice', 
         f"{avg_nb['dice_mean']:.4f}",
         f"{avg_crf['dice_mean']:.4f}",
         f"{improvements['dice_mean']:+.4f}"]
    ]
    
    table = ax.table(cellText=metrics_data,
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.35, 0.25, 0.25, 0.25])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    for i in range(4):
        table[(0, i)].set_facecolor('#2196F3')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, len(metrics_data)):
        improvement_val = float(metrics_data[i][3])
        if improvement_val > 0:
            table[(i, 3)].set_facecolor('#c8e6c9')
        elif improvement_val < 0:
            table[(i, 3)].set_facecolor('#ffcdd2')
        
        if i % 2 == 0:
            for j in range(3):
                table[(i, j)].set_facecolor('#f5f5f5')
    
    # Highlight mean dice row
    for j in range(4):
        table[(5, j)].set_facecolor('#fff9c4')
        table[(5, j)].set_text_props(weight='bold')
    
    n_images = len(metrics_crf_list)
    plt.title(f'Overall Summary - {n_images} Test Images',
              fontsize=16, fontweight='bold', pad=20)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_crf_comparison(original, naive_mask, crf_mask, gt_mask=None, save_path=None):
    """Visualize comparison between Naive Bayes and CRF refinement"""
    if gt_mask is not None:
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(gt_mask, cmap='gray')
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
        axes[2].imshow(naive_mask, cmap='gray')
        axes[2].set_title('Naive Bayes')
        axes[2].axis('off')
        
        axes[3].imshow(crf_mask, cmap='gray')
        axes[3].set_title('With CRF')
        axes[3].axis('off')
    else:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(naive_mask, cmap='gray')
        axes[1].set_title('Naive Bayes')
        axes[1].axis('off')
        
        axes[2].imshow(crf_mask, cmap='gray')
        axes[2].set_title('With CRF')
        axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.close(fig)


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Memory-efficient main execution pipeline with CRF"""
    
    # Configuration
    TRAIN_FUNDUS_DIR = 'REFUGE2\\Train\\REFUGE1-train\\Training400\\Non-Glaucoma'
    TRAIN_MASK_DIR = 'REFUGE2\\Train\\REFUGE1-train\\Disc_Cup_Masks\\Non-Glaucoma'
    TEST_FUNDUS_DIR = 'REFUGE2\\Test\\refuge2-test'
    TEST_MASK_DIR = 'REFUGE2\\Test\\Disc_Mask'
    OUTPUT_DIR = 'CRFOutput'
    
    USE_BOTH_COLORSPACES = True
    APPLY_CRF = True
    
    # MEMORY OPTIMIZATION SETTINGS
    MAX_IMAGE_SIZE = 512      # Resize images to max 512x512
    BATCH_SIZE = 10            # Process 3 images at a time
    MAX_TRAIN_IMAGES = 400     # Limit training images
    MAX_TEST_IMAGES = 25      # Limit test images
    
    # CRF parameters (reduced for memory efficiency)
    CRF_PARAMS = {
        'n_iters': 5,
        'sxy_gaussian': 3,
        'compat_gaussian': 3,
        'sxy_bilateral': 60,
        'srgb_bilateral': 10,
        'compat_bilateral': 10
    }
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # ========== TRAINING ==========
    print("="*60)
    print("TRAINING PHASE (MEMORY EFFICIENT)")
    print("="*60)
    
    fundus_files = sorted(glob(os.path.join(TRAIN_FUNDUS_DIR, '*.*')))[:MAX_TRAIN_IMAGES]
    mask_files = sorted(glob(os.path.join(TRAIN_MASK_DIR, '*.*')))[:MAX_TRAIN_IMAGES]
    
    print(f"Found {len(fundus_files)} training images")
    
    clf, scaler = train_naive_bayes_memory_efficient(
        fundus_files, mask_files,
        use_both_colorspaces=USE_BOTH_COLORSPACES,
        batch_size=BATCH_SIZE,
        max_size=MAX_IMAGE_SIZE
    )
    
    gc.collect()
    
    # ========== TESTING ==========
    print("\n" + "="*60)
    print("TESTING PHASE WITH CRF REFINEMENT")
    print("="*60)
    
    test_files = sorted(glob(os.path.join(TEST_FUNDUS_DIR, '*.*')))[:MAX_TEST_IMAGES]
    print(f"Found {len(test_files)} test images")
    
    metrics_nb = []
    metrics_crf = []
    
    for i, test_file in enumerate(test_files):
        print(f"\nProcessing test image {i+1}/{len(test_files)}: {os.path.basename(test_file)}")
        
        test_image = cv2.imread(test_file)
        if test_image is None:
            continue
        
        # Resize test image
        test_image = resize_image(test_image, max_size=MAX_IMAGE_SIZE)
        
        # Predict with CRF
        crf_mask, nb_mask = predict_with_crf(
            clf, scaler, test_image,
            use_both_colorspaces=USE_BOTH_COLORSPACES,
            apply_crf=APPLY_CRF,
            crf_params=CRF_PARAMS
        )
        
        # Evaluate if ground truth exists
        test_basename = os.path.basename(test_file)
        base_name = os.path.splitext(test_basename)[0]
        gt_file = os.path.join(TEST_MASK_DIR, test_basename[:-3]+'png')
        
        if os.path.exists(gt_file):
            gt_mask = cv2.imread(gt_file, cv2.IMREAD_GRAYSCALE)
            gt_mask = resize_mask(gt_mask, max_size=MAX_IMAGE_SIZE)
            gt_mask = remap_mask_labels(gt_mask)
            
            # Evaluate both methods
            metrics_nb_img = evaluate_segmentation(nb_mask, gt_mask)
            metrics_crf_img = evaluate_segmentation(crf_mask, gt_mask)
            
            metrics_nb.append(metrics_nb_img)
            metrics_crf.append(metrics_crf_img)
            
            print(f"  Naive Bayes - Accuracy: {metrics_nb_img['accuracy']:.4f}, Dice: {metrics_nb_img['dice_mean']:.4f}")
            print(f"  With CRF    - Accuracy: {metrics_crf_img['accuracy']:.4f}, Dice: {metrics_crf_img['dice_mean']:.4f}")
            print(f"  Improvement: {(metrics_crf_img['dice_mean'] - metrics_nb_img['dice_mean']):+.4f}")
            
            # Save comparison metrics
            comparison_path = os.path.join(OUTPUT_DIR, f"{base_name}_metrics_comparison.png")
            save_comparison_metrics_as_image(metrics_nb_img, metrics_crf_img, 
                                            base_name, comparison_path)
        else:
            gt_mask = None
            print("  No ground truth available")
        
        # Visualize comparison
        viz_path = os.path.join(OUTPUT_DIR, f"{base_name}_comparison.png")
        visualize_crf_comparison(test_image, nb_mask, crf_mask, gt_mask, viz_path)
        
        # Clear memory
        plt.close('all')
        del test_image, crf_mask, nb_mask
        if gt_mask is not None:
            del gt_mask
        gc.collect()
    
    # ========== SUMMARY ==========
    if metrics_crf:
        print("\n" + "="*60)
        print("OVERALL COMPARISON")
        print("="*60)
        
        print("\nNaive Bayes:")
        print(f"  Avg Accuracy: {np.mean([m['accuracy'] for m in metrics_nb]):.4f}")
        print(f"  Avg Dice: {np.mean([m['dice_mean'] for m in metrics_nb]):.4f}")
        
        print("\nWith CRF Refinement:")
        print(f"  Avg Accuracy: {np.mean([m['accuracy'] for m in metrics_crf]):.4f}")
        print(f"  Avg Dice: {np.mean([m['dice_mean'] for m in metrics_crf]):.4f}")
        
        improvement = (np.mean([m['dice_mean'] for m in metrics_crf]) - 
                      np.mean([m['dice_mean'] for m in metrics_nb]))
        print(f"\nOverall Dice Improvement: {improvement:+.4f}")
        
        # Save summary metrics
        summary_path = os.path.join(OUTPUT_DIR, "overall_metrics_summary.png")
        save_summary_metrics_as_image(metrics_nb, metrics_crf, summary_path)
        print(f"\nSummary metrics saved to: {summary_path}")
    
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()
