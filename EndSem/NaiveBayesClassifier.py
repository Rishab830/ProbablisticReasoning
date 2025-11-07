import numpy as np
import cv2
import os
import gc
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
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
        # Use INTER_NEAREST to avoid creating intermediate values
        resized = cv2.resize(mask, (new_width, new_height), 
                           interpolation=cv2.INTER_NEAREST)
        return resized
    return mask


def remap_mask_labels(mask):
    """
    Remap mask values to exactly 0, 128, or 255
    Handles cases where resizing creates intermediate values
    """
    output = np.zeros_like(mask)
    output[mask < 64] = 0          # 0-63 -> 0 (disc)
    output[(mask >= 64) & (mask < 192)] = 128  # 64-191 -> 128 (cup)
    output[mask >= 192] = 255      # 192-255 -> 255 (background)
    return output


# ============================================================================
# FEATURE EXTRACTION FUNCTIONS
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
# MEMORY-EFFICIENT TRAINING FUNCTION
# ============================================================================

def train_naive_bayes_memory_efficient(fundus_files, mask_files, 
                                       use_both_colorspaces=True, 
                                       batch_size=5, max_size=512):
    """
    Memory-efficient training with batching and incremental learning
    
    Parameters:
    -----------
    fundus_files : list
        List of paths to fundus images
    mask_files : list
        List of paths to mask images
    use_both_colorspaces : bool
        Whether to use both RGB and HSV features
    batch_size : int
        Number of images to process at once
    max_size : int
        Maximum image dimension for resizing
    
    Returns:
    --------
    clf : GaussianNB
        Trained classifier
    scaler : StandardScaler
        Fitted feature scaler
    """
    print("Training with memory-efficient batching...")
    scaler = StandardScaler()
    clf = GaussianNB()
    
    # Process in batches
    n_batches = (len(fundus_files) + batch_size - 1) // batch_size
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(fundus_files))
        
        print(f"Processing batch {batch_idx + 1}/{n_batches} (images {start_idx+1}-{end_idx})...")
        
        X_batch = []
        y_batch = []
        
        for i in range(start_idx, end_idx):
            # Load images
            fundus = cv2.imread(fundus_files[i])
            mask = cv2.imread(mask_files[i], cv2.IMREAD_GRAYSCALE)
            
            if fundus is None or mask is None:
                print(f"  Warning: Could not load image pair {i+1}")
                continue
            
            # Resize - use different methods for fundus and mask
            fundus = resize_image(fundus, max_size=max_size)
            mask = resize_mask(mask, max_size=max_size)
            
            # Remap mask values to ensure only 0, 128, 255
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
            
            # Clear references
            del fundus, mask
        
        if len(X_batch) == 0:
            print(f"  Skipping batch {batch_idx + 1} - no valid images")
            continue
            
        X_batch = np.vstack(X_batch)
        y_batch = np.concatenate(y_batch)
        
        print(f"  Batch samples: {X_batch.shape[0]}, Features: {X_batch.shape[1]}")
        
        # Partial fit for incremental learning
        if batch_idx == 0:
            # First batch: fit scaler and initialize classifier
            X_batch_scaled = scaler.fit_transform(X_batch)
            clf.partial_fit(X_batch_scaled, y_batch, classes=np.array([0, 128, 255]))
        else:
            X_batch_scaled = scaler.transform(X_batch)
            clf.partial_fit(X_batch_scaled, y_batch)
        
        # Clear batch data and force garbage collection
        del X_batch, y_batch, X_batch_scaled
        gc.collect()
    
    print("Training complete!")
    return clf, scaler


# ============================================================================
# PREDICTION FUNCTION
# ============================================================================

def predict_segmentation(clf, scaler, test_image, use_both_colorspaces=True):
    """Predict segmentation mask for a test image"""
    # Extract features
    features = extract_features_with_coords(
        test_image,
        normalize_coords=True,
        use_both_colorspaces=use_both_colorspaces
    )
    
    # Standardize features
    features_scaled = scaler.transform(features)
    
    # Predict
    predictions = clf.predict(features_scaled)
    segmented_mask = predictions.reshape(test_image.shape[:2])
    
    return segmented_mask


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def calculate_dice_coefficient(pred, gt, label):
    """Calculate Dice coefficient for a specific class"""
    pred_label = (pred == label).astype(float)
    gt_label = (gt == label).astype(float)
    
    intersection = np.sum(pred_label * gt_label)
    dice = (2.0 * intersection) / (np.sum(pred_label) + np.sum(gt_label) + 1e-8)
    
    return dice


def calculate_iou(pred, gt, label):
    """Calculate Intersection over Union (IoU) for a specific class"""
    pred_label = (pred == label).astype(float)
    gt_label = (gt == label).astype(float)
    
    intersection = np.sum(pred_label * gt_label)
    union = np.sum(pred_label) + np.sum(gt_label) - intersection
    iou = intersection / (union + 1e-8)
    
    return iou


def evaluate_segmentation(pred_mask, gt_mask):
    """Evaluate segmentation performance"""
    # Overall pixel accuracy
    accuracy = accuracy_score(gt_mask.flatten(), pred_mask.flatten())
    
    # Per-class metrics
    dice_disc = calculate_dice_coefficient(pred_mask, gt_mask, 0)
    dice_cup = calculate_dice_coefficient(pred_mask, gt_mask, 128)
    dice_bg = calculate_dice_coefficient(pred_mask, gt_mask, 255)
    
    iou_disc = calculate_iou(pred_mask, gt_mask, 0)
    iou_cup = calculate_iou(pred_mask, gt_mask, 128)
    iou_bg = calculate_iou(pred_mask, gt_mask, 255)
    
    metrics = {
        'accuracy': accuracy,
        'dice_disc': dice_disc,
        'dice_cup': dice_cup,
        'dice_background': dice_bg,
        'dice_mean': (dice_disc + dice_cup + dice_bg) / 3,
        'iou_disc': iou_disc,
        'iou_cup': iou_cup,
        'iou_background': iou_bg,
        'iou_mean': (iou_disc + iou_cup + iou_bg) / 3
    }
    
    return metrics


# ============================================================================
# METRICS VISUALIZATION AND SAVING
# ============================================================================

def save_metrics_as_image(metrics_dict, image_name, save_path):
    """
    Save evaluation metrics as a table image
    
    Parameters:
    -----------
    metrics_dict : dict
        Dictionary containing evaluation metrics
    image_name : str
        Name of the image being evaluated
    save_path : str
        Full path where to save the metrics image
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data for table
    metrics_data = [
        ['Metric', 'Value'],                                    # Row 0
        ['Pixel Accuracy', f"{metrics_dict['accuracy']:.4f}"],  # Row 1
        ['', ''],                                               # Row 2 (separator)
        ['Dice - Disc (0)', f"{metrics_dict['dice_disc']:.4f}"],        # Row 3
        ['Dice - Cup (128)', f"{metrics_dict['dice_cup']:.4f}"],        # Row 4
        ['Dice - Background (255)', f"{metrics_dict['dice_background']:.4f}"],  # Row 5
        ['Mean Dice Score', f"{metrics_dict['dice_mean']:.4f}"],        # Row 6
        ['', ''],                                               # Row 7 (separator)
        ['IoU - Disc (0)', f"{metrics_dict['iou_disc']:.4f}"], # Row 8
        ['IoU - Cup (128)', f"{metrics_dict['iou_cup']:.4f}"], # Row 9
        ['IoU - Background (255)', f"{metrics_dict['iou_background']:.4f}"],  # Row 10
        ['Mean IoU', f"{metrics_dict['iou_mean']:.4f}"]        # Row 11
    ]
    
    # Create table
    table = ax.table(cellText=metrics_data, 
                     cellLoc='left',
                     loc='center',
                     colWidths=[0.6, 0.3])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2.2)
    
    # Style header row (row 0)
    for i in range(2):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style separator rows (rows 2 and 7)
    for sep_row in [2, 7]:
        for j in range(2):
            table[(sep_row, j)].set_facecolor('#e0e0e0')
    
    # Alternate row colors (excluding separators and mean rows)
    for i in [1, 3, 5, 8, 10]:  # Changed: removed 11, added proper rows
        for j in range(2):
            table[(i, j)].set_facecolor('#f9f9f9')
    
    # Highlight mean scores (rows 6 and 11)
    for mean_row in [6, 11]:  # Changed from [7, 12] to [6, 11]
        for j in range(2):
            table[(mean_row, j)].set_facecolor('#fff9c4')
            table[(mean_row, j)].set_text_props(weight='bold')
    
    # Add title
    plt.title(f'Segmentation Metrics - {image_name}', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Save figure
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def save_summary_metrics_as_image(all_metrics, save_path):
    """
    Save overall summary metrics as a table image
    
    Parameters:
    -----------
    all_metrics : list of dict
        List of metrics dictionaries from all test images
    save_path : str
        Path to save the summary image
    """
    if not all_metrics:
        return
    
    # Calculate mean and std for each metric
    mean_metrics = {
        key: np.mean([m[key] for m in all_metrics])
        for key in all_metrics[0].keys()
    }
    
    std_metrics = {
        key: np.std([m[key] for m in all_metrics])
        for key in all_metrics[0].keys()
    }
    
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data
    metrics_data = [
        ['Metric', 'Mean', 'Std Dev'],                          # Row 0
        ['Pixel Accuracy',                                      # Row 1
         f"{mean_metrics['accuracy']:.4f}",
         f"{std_metrics['accuracy']:.4f}"],
        ['', '', ''],                                           # Row 2 (separator)
        ['Dice - Disc (0)',                                     # Row 3
         f"{mean_metrics['dice_disc']:.4f}",
         f"{std_metrics['dice_disc']:.4f}"],
        ['Dice - Cup (128)',                                    # Row 4
         f"{mean_metrics['dice_cup']:.4f}",
         f"{std_metrics['dice_cup']:.4f}"],
        ['Dice - Background (255)',                             # Row 5
         f"{mean_metrics['dice_background']:.4f}",
         f"{std_metrics['dice_background']:.4f}"],
        ['Mean Dice Score',                                     # Row 6
         f"{mean_metrics['dice_mean']:.4f}",
         f"{std_metrics['dice_mean']:.4f}"],
        ['', '', ''],                                           # Row 7 (separator)
        ['IoU - Disc (0)',                                      # Row 8
         f"{mean_metrics['iou_disc']:.4f}",
         f"{std_metrics['iou_disc']:.4f}"],
        ['IoU - Cup (128)',                                     # Row 9
         f"{mean_metrics['iou_cup']:.4f}",
         f"{std_metrics['iou_cup']:.4f}"],
        ['IoU - Background (255)',                              # Row 10
         f"{mean_metrics['iou_background']:.4f}",
         f"{std_metrics['iou_background']:.4f}"],
        ['Mean IoU',                                            # Row 11
         f"{mean_metrics['iou_mean']:.4f}",
         f"{std_metrics['iou_mean']:.4f}"]
    ]
    
    # Create table
    table = ax.table(cellText=metrics_data,
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.5, 0.25, 0.25])
    
    # Style
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.2)
    
    # Header styling (row 0)
    for i in range(3):
        table[(0, i)].set_facecolor('#2196F3')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style separator rows (rows 2 and 7)
    for sep_row in [2, 7]:
        for j in range(3):
            table[(sep_row, j)].set_facecolor('#e0e0e0')
    
    # Alternate row colors (excluding separators and mean rows)
    for i in [1, 3, 5, 8, 10]:  # Changed: removed 11, added correct rows
        for j in range(3):
            table[(i, j)].set_facecolor('#f9f9f9')
    
    # Highlight mean scores (rows 6 and 11)
    for mean_row in [6, 11]:  # Changed from [7, 12] to [6, 11]
        for j in range(3):
            table[(mean_row, j)].set_facecolor('#fff9c4')
            table[(mean_row, j)].set_text_props(weight='bold')
    
    # Add title with sample count
    n_images = len(all_metrics)
    plt.title(f'Overall Summary - {n_images} Test Images',
              fontsize=16, fontweight='bold', pad=20)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def visualize_results(original, predicted_mask, gt_mask=None, save_path=None):
    """Visualize segmentation results"""
    if gt_mask is not None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(gt_mask, cmap='gray')
        axes[1].set_title('Ground Truth Mask')
        axes[1].axis('off')
        
        axes[2].imshow(predicted_mask, cmap='gray')
        axes[2].set_title('Predicted Mask')
        axes[2].axis('off')
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(predicted_mask, cmap='gray')
        axes[1].set_title('Predicted Mask')
        axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    # Close figure to free memory
    plt.close(fig)


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Memory-efficient main execution pipeline"""
    
    # ========== CONFIGURATION ==========
    TRAIN_FUNDUS_DIR = 'REFUGE2\\Train\\REFUGE1-train\\Training400\\Non-Glaucoma'
    TRAIN_MASK_DIR = 'REFUGE2\\Train\\REFUGE1-train\\Disc_Cup_Masks\\Non-Glaucoma'
    TEST_FUNDUS_DIR = 'REFUGE2\\Test\\refuge2-test'
    TEST_MASK_DIR = 'REFUGE2\\Test\\Disc_Mask'
    OUTPUT_DIR = 'NaiveBayesClassifierOutput'
    
    USE_BOTH_COLORSPACES = True
    
    # MEMORY OPTIMIZATION SETTINGS
    MAX_IMAGE_SIZE = 512      # Resize images to max 512x512
    BATCH_SIZE = 10            # Process 5 images at a time during training
    MAX_TRAIN_IMAGES = 400     # Limit training images
    MAX_TEST_IMAGES = 25      # Limit test images
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # ========== TRAINING ==========
    print("="*60)
    print("TRAINING PHASE (MEMORY EFFICIENT)")
    print("="*60)
    
    # Get training file paths
    fundus_files = sorted(glob(os.path.join(TRAIN_FUNDUS_DIR, '*.*')))[:MAX_TRAIN_IMAGES]
    mask_files = sorted(glob(os.path.join(TRAIN_MASK_DIR, '*.*')))[:MAX_TRAIN_IMAGES]
    
    print(f"Found {len(fundus_files)} training images")
    
    # Train with memory-efficient batching
    clf, scaler = train_naive_bayes_memory_efficient(
        fundus_files, 
        mask_files,
        use_both_colorspaces=USE_BOTH_COLORSPACES,
        batch_size=BATCH_SIZE,
        max_size=MAX_IMAGE_SIZE
    )
    
    # Clear memory after training
    gc.collect()
    
    # ========== TESTING ==========
    print("\n" + "="*60)
    print("TESTING PHASE")
    print("="*60)
    
    # Get test images
    test_files = sorted(glob(os.path.join(TEST_FUNDUS_DIR, '*.*')))[:MAX_TEST_IMAGES]
    print(f"Found {len(test_files)} test images")
    
    all_metrics = []
    
    for i, test_file in enumerate(test_files):
        print(f"\nProcessing test image {i+1}/{len(test_files)}: {os.path.basename(test_file)}")
        
        # Load and resize test image
        test_image = cv2.imread(test_file)
        
        if test_image is None:
            print(f"  Could not load {test_file}")
            continue
        
        # Resize test image
        test_image = resize_image(test_image, max_size=MAX_IMAGE_SIZE)
        
        # Predict segmentation
        print("  Predicting segmentation...")
        predicted_mask = predict_segmentation(
            clf, scaler, test_image,
            use_both_colorspaces=USE_BOTH_COLORSPACES
        )
        
        # Load and process ground truth if available
        test_basename = os.path.basename(test_file)
        base_name = os.path.splitext(test_basename)[0]
        gt_file = os.path.join(TEST_MASK_DIR, test_basename[:-3]+'png')
        
        if os.path.exists(gt_file):
            gt_mask = cv2.imread(gt_file, cv2.IMREAD_GRAYSCALE)
            gt_mask = resize_mask(gt_mask, max_size=MAX_IMAGE_SIZE)
            gt_mask = remap_mask_labels(gt_mask)
            
            # Evaluate
            metrics = evaluate_segmentation(predicted_mask, gt_mask)
            all_metrics.append(metrics)
            
            print(f"  Pixel Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Mean Dice: {metrics['dice_mean']:.4f}")
            print(f"  Mean IoU: {metrics['iou_mean']:.4f}")
            
            # Save metrics as image
            metrics_path = os.path.join(OUTPUT_DIR, f"{base_name}_metrics.png")
            save_metrics_as_image(metrics, base_name, metrics_path)
        else:
            gt_mask = None
            print("  No ground truth available")
        
        # Visualize and save comparison
        viz_save_path = os.path.join(OUTPUT_DIR, f"{base_name}_comparison.png")
        visualize_results(test_image, predicted_mask, gt_mask, viz_save_path)
        
        # Clear memory after each image
        plt.close('all')
        del test_image, predicted_mask
        if gt_mask is not None:
            del gt_mask
        gc.collect()
    
    # ========== SUMMARY ==========
    if all_metrics:
        print("\n" + "="*60)
        print("OVERALL EVALUATION SUMMARY")
        print("="*60)
        
        mean_metrics = {
            key: np.mean([m[key] for m in all_metrics])
            for key in all_metrics[0].keys()
        }
        
        print(f"Average Pixel Accuracy: {mean_metrics['accuracy']:.4f}")
        print(f"Average Mean Dice Score: {mean_metrics['dice_mean']:.4f}")
        print(f"  - Disc Dice: {mean_metrics['dice_disc']:.4f}")
        print(f"  - Cup Dice: {mean_metrics['dice_cup']:.4f}")
        print(f"  - Background Dice: {mean_metrics['dice_background']:.4f}")
        print(f"Average Mean IoU: {mean_metrics['iou_mean']:.4f}")
        print(f"  - Disc IoU: {mean_metrics['iou_disc']:.4f}")
        print(f"  - Cup IoU: {mean_metrics['iou_cup']:.4f}")
        print(f"  - Background IoU: {mean_metrics['iou_background']:.4f}")
        
        # Save summary metrics as image
        summary_path = os.path.join(OUTPUT_DIR, "overall_summary_metrics.png")
        save_summary_metrics_as_image(all_metrics, summary_path)
        print(f"\nSummary metrics saved to: {summary_path}")
    
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()
