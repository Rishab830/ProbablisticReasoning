import numpy as np
import cv2
import os
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from glob import glob

# ============================================================================
# FEATURE EXTRACTION FUNCTIONS
# ============================================================================

def extract_features_with_coords(image, normalize_coords=True, use_both_colorspaces=True):
    """
    Extract color and spatial features from fundus image
    
    Parameters:
    -----------
    image : numpy array
        Input fundus image (BGR format from cv2)
    normalize_coords : bool
        Whether to normalize coordinates to [0,1]
    use_both_colorspaces : bool
        If True, use both RGB and HSV; if False, use only RGB
    
    Returns:
    --------
    features : numpy array
        Feature matrix of shape (n_pixels, n_features)
    """
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
# DATA LOADING FUNCTIONS
# ============================================================================

def load_training_data(fundus_dir, mask_dir, max_images=None):
    """
    Load fundus images and corresponding masks from directories
    
    Parameters:
    -----------
    fundus_dir : str
        Directory containing fundus images
    mask_dir : str
        Directory containing mask images
    max_images : int or None
        Maximum number of images to load (None for all)
    
    Returns:
    --------
    fundus_images : list
        List of fundus images
    mask_images : list
        List of corresponding masks
    """
    # Get sorted list of image files
    fundus_files = sorted(glob(os.path.join(fundus_dir, '*.*')))
    mask_files = sorted(glob(os.path.join(mask_dir, '*.*')))
    
    if max_images:
        fundus_files = fundus_files[:max_images]
        mask_files = mask_files[:max_images]
    
    fundus_images = []
    mask_images = []
    
    print(f"Loading {len(fundus_files)} training images...")
    for fundus_file, mask_file in zip(fundus_files, mask_files):
        fundus = cv2.imread(fundus_file)
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        
        if fundus is not None and mask is not None:
            fundus_images.append(fundus)
            mask_images.append(mask)
        else:
            print(f"Warning: Could not load {fundus_file} or {mask_file}")
    
    print(f"Successfully loaded {len(fundus_images)} image pairs")
    return fundus_images, mask_images


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_naive_bayes(fundus_images, mask_images, use_both_colorspaces=True):
    """
    Train Gaussian Naive Bayes classifier on fundus images
    
    Parameters:
    -----------
    fundus_images : list
        List of training fundus images
    mask_images : list
        List of training mask images
    use_both_colorspaces : bool
        Whether to use both RGB and HSV features
    
    Returns:
    --------
    clf : GaussianNB
        Trained classifier
    scaler : StandardScaler
        Fitted feature scaler
    """
    print("Extracting features from training images...")
    X_train = []
    y_train = []
    
    for i, (fundus, mask) in enumerate(zip(fundus_images, mask_images)):
        print(f"Processing training image {i+1}/{len(fundus_images)}...")
        
        # Extract features
        features = extract_features_with_coords(
            fundus, 
            normalize_coords=True,
            use_both_colorspaces=use_both_colorspaces
        )
        labels = mask.flatten()
        
        X_train.append(features)
        y_train.append(labels)
    
    # Combine all training data
    X_train = np.vstack(X_train)
    y_train = np.concatenate(y_train)
    
    print(f"Total training samples: {X_train.shape[0]}")
    print(f"Feature dimensions: {X_train.shape[1]}")
    print(f"Class distribution: {np.unique(y_train, return_counts=True)}")
    
    # Standardize features
    print("Standardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train classifier
    print("Training Naive Bayes classifier...")
    clf = GaussianNB()
    clf.fit(X_train_scaled, y_train)
    
    print("Training complete!")
    return clf, scaler


# ============================================================================
# PREDICTION FUNCTION
# ============================================================================

def predict_segmentation(clf, scaler, test_image, use_both_colorspaces=True):
    """
    Predict segmentation mask for a test image
    
    Parameters:
    -----------
    clf : GaussianNB
        Trained classifier
    scaler : StandardScaler
        Fitted feature scaler
    test_image : numpy array
        Test fundus image
    use_both_colorspaces : bool
        Whether to use both RGB and HSV features
    
    Returns:
    --------
    segmented_mask : numpy array
        Predicted segmentation mask
    """
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
    """
    Evaluate segmentation performance
    
    Parameters:
    -----------
    pred_mask : numpy array
        Predicted segmentation mask
    gt_mask : numpy array
        Ground truth mask
    
    Returns:
    --------
    metrics : dict
        Dictionary containing evaluation metrics
    """
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
# VISUALIZATION AND SAVING FUNCTIONS
# ============================================================================

def visualize_results(original, predicted_mask, gt_mask=None, save_path=None):
    """
    Visualize segmentation results
    
    Parameters:
    -----------
    original : numpy array
        Original fundus image
    predicted_mask : numpy array
        Predicted segmentation mask
    gt_mask : numpy array or None
        Ground truth mask (if available)
    save_path : str or None
        Path to save the visualization
    """
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
        print(f"Visualization saved to {save_path}")
    
    # plt.show()


def save_mask(mask, save_path):
    """Save segmentation mask as image"""
    cv2.imwrite(save_path, mask.astype(np.uint8))
    print(f"Mask saved to {save_path}")


def create_colored_overlay(original, mask):
    """
    Create colored overlay of segmentation on original image
    
    Parameters:
    -----------
    original : numpy array
        Original fundus image
    mask : numpy array
        Segmentation mask (values: 0, 128, 255)
    
    Returns:
    --------
    overlay : numpy array
        Image with colored segmentation overlay
    """
    overlay = original.copy()
    
    # Create colored masks
    disc_mask = (mask == 0)
    cup_mask = (mask == 128)
    
    # Apply colors with transparency
    overlay[disc_mask] = cv2.addWeighted(
        overlay[disc_mask], 0.6,
        np.full_like(overlay[disc_mask], [0, 255, 0]), 0.4,  # Green for disc
        0
    )
    overlay[cup_mask] = cv2.addWeighted(
        overlay[cup_mask], 0.6,
        np.full_like(overlay[cup_mask], [255, 0, 0]), 0.4,  # Blue for cup
        0
    )
    
    return overlay


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main execution pipeline"""
    
    # ========== CONFIGURATION ==========
    TRAIN_FUNDUS_DIR = 'REFUGE2\\Train\\REFUGE1-train\\Training400\\Glaucoma'
    TRAIN_MASK_DIR = 'REFUGE2\\Train\\REFUGE1-train\\Disc_Cup_Masks\\Glaucoma'
    TEST_FUNDUS_DIR = 'REFUGE2\\Test\\refuge2-test'
    TEST_MASK_DIR = 'REFUGE2\\Test\\Disc_Mask'  # Optional, for evaluation
    OUTPUT_DIR = 'NaiveBayesClassifierOutput'
    
    USE_BOTH_COLORSPACES = True  # Set to False to use only RGB
    MAX_TRAIN_IMAGES = 20  # Set to a number to limit training data
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # ========== TRAINING ==========
    print("="*60)
    print("TRAINING PHASE")
    print("="*60)
    
    # Load training data
    fundus_train, masks_train = load_training_data(
        TRAIN_FUNDUS_DIR, 
        TRAIN_MASK_DIR,
        max_images=MAX_TRAIN_IMAGES
    )
    
    # Train classifier
    clf, scaler = train_naive_bayes(
        fundus_train, 
        masks_train,
        use_both_colorspaces=USE_BOTH_COLORSPACES
    )
    
    # ========== TESTING ==========
    print("\n" + "="*60)
    print("TESTING PHASE")
    print("="*60)
    
    # Get test images
    test_files = sorted(glob(os.path.join(TEST_FUNDUS_DIR, '*.*')))
    test_files = test_files[:11]
    all_metrics = []
    
    for i, test_file in enumerate(test_files):
        print(f"\nProcessing test image {i+1}/{len(test_files)}: {os.path.basename(test_file)}")
        
        # Load test image
        test_image = cv2.imread(test_file)
        
        if test_image is None:
            print(f"Could not load {test_file}")
            continue
        
        # Predict segmentation
        print("Predicting segmentation...")
        predicted_mask = predict_segmentation(
            clf, scaler, test_image,
            use_both_colorspaces=USE_BOTH_COLORSPACES
        )
        
        # Load ground truth if available
        test_basename = os.path.basename(test_file)
        gt_file = os.path.join(TEST_MASK_DIR, test_basename[:-3]+'png')
        
        if os.path.exists(gt_file):
            gt_mask = cv2.imread(gt_file, cv2.IMREAD_GRAYSCALE)
            
            # Evaluate
            metrics = evaluate_segmentation(predicted_mask, gt_mask)
            all_metrics.append(metrics)
            
            print(f"Pixel Accuracy: {metrics['accuracy']:.4f}")
            print(f"Mean Dice Score: {metrics['dice_mean']:.4f}")
            print(f"  - Disc Dice: {metrics['dice_disc']:.4f}")
            print(f"  - Cup Dice: {metrics['dice_cup']:.4f}")
            print(f"  - Background Dice: {metrics['dice_background']:.4f}")
            print(f"Mean IoU: {metrics['iou_mean']:.4f}")
        else:
            gt_mask = None
            print("No ground truth available for evaluation")
        
        # Save results
        base_name = os.path.splitext(test_basename)[0]
        
        # Save predicted mask
        mask_save_path = os.path.join(OUTPUT_DIR, f"{base_name}_predicted_mask.png")
        save_mask(predicted_mask, mask_save_path)
        
        # Save colored overlay
        overlay = create_colored_overlay(test_image, predicted_mask)
        overlay_save_path = os.path.join(OUTPUT_DIR, f"{base_name}_overlay.png")
        cv2.imwrite(overlay_save_path, overlay)
        print(f"Overlay saved to {overlay_save_path}")
        
        # Visualize and save comparison
        viz_save_path = os.path.join(OUTPUT_DIR, f"{base_name}_comparison.png")
        visualize_results(test_image, predicted_mask, gt_mask, viz_save_path)
    
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
    
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
