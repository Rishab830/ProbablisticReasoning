import numpy as np
import cv2
import os
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian
import matplotlib.pyplot as plt
from glob import glob

# ============================================================================
# FEATURE EXTRACTION (Same as before)
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
# TRAINING (Same as before)
# ============================================================================

def train_naive_bayes(fundus_images, mask_images, use_both_colorspaces=True):
    """Train Gaussian Naive Bayes classifier"""
    print("Extracting features from training images...")
    X_train = []
    y_train = []
    
    for i, (fundus, mask) in enumerate(zip(fundus_images, mask_images)):
        print(f"Processing training image {i+1}/{len(fundus_images)}...")
        
        features = extract_features_with_coords(
            fundus, 
            normalize_coords=True,
            use_both_colorspaces=use_both_colorspaces
        )
        labels = mask.flatten()
        
        X_train.append(features)
        y_train.append(labels)
    
    X_train = np.vstack(X_train)
    y_train = np.concatenate(y_train)
    
    print(f"Total training samples: {X_train.shape[0]}")
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train classifier
    print("Training Naive Bayes classifier...")
    clf = GaussianNB()
    clf.fit(X_train_scaled, y_train)
    
    return clf, scaler


# ============================================================================
# CRF REFINEMENT FUNCTIONS
# ============================================================================

def get_probability_map(clf, scaler, image, use_both_colorspaces=True):
    """
    Get probability map from Naive Bayes classifier
    
    Returns:
    --------
    probabilities : numpy array
        Shape (n_classes, height, width) with probability for each class
    """
    height, width = image.shape[:2]
    
    # Extract features
    features = extract_features_with_coords(
        image,
        normalize_coords=True,
        use_both_colorspaces=use_both_colorspaces
    )
    
    # Standardize
    features_scaled = scaler.transform(features)
    
    # Get probabilities for each class
    prob = clf.predict_proba(features_scaled)  # Shape: (n_pixels, n_classes)
    
    # Reshape to (n_classes, height, width)
    n_classes = prob.shape[1]
    prob = prob.T.reshape(n_classes, height, width)
    
    return prob


def apply_dense_crf(image, probabilities, n_iters=5, 
                    sxy_gaussian=3, compat_gaussian=3,
                    sxy_bilateral=80, srgb_bilateral=13, compat_bilateral=10):
    """
    Apply Dense CRF refinement to segmentation probabilities
    
    Parameters:
    -----------
    image : numpy array
        Original RGB image (H, W, 3)
    probabilities : numpy array
        Class probabilities from classifier (n_classes, H, W)
    n_iters : int
        Number of CRF iterations
    sxy_gaussian : float
        Spatial standard deviation for Gaussian kernel
    compat_gaussian : float
        Compatibility for Gaussian kernel
    sxy_bilateral : float
        Spatial standard deviation for bilateral kernel
    srgb_bilateral : float
        Color standard deviation for bilateral kernel
    compat_bilateral : float
        Compatibility for bilateral kernel
    
    Returns:
    --------
    refined_mask : numpy array
        Refined segmentation mask (H, W)
    """
    height, width = image.shape[:2]
    n_classes = probabilities.shape[0]
    
    # Create DenseCRF object
    d = dcrf.DenseCRF2D(width, height, n_classes)
    
    # Set unary potentials (negative log probability)
    unary = unary_from_softmax(probabilities)
    unary = np.ascontiguousarray(unary)
    d.setUnaryEnergy(unary)
    
    # Create pairwise potentials
    # 1. Gaussian kernel: encourages nearby pixels to have same label
    d.addPairwiseGaussian(sxy=sxy_gaussian, compat=compat_gaussian,
                          kernel=dcrf.DIAG_KERNEL,
                          normalization=dcrf.NORMALIZE_SYMMETRIC)
    
    # 2. Bilateral kernel: encourages pixels with similar color and position to have same label
    d.addPairwiseBilateral(sxy=sxy_bilateral, srgb=srgb_bilateral,
                          rgbim=image.astype(np.uint8),
                          compat=compat_bilateral,
                          kernel=dcrf.DIAG_KERNEL,
                          normalization=dcrf.NORMALIZE_SYMMETRIC)
    
    # Inference
    Q = d.inference(n_iters)
    
    # Get MAP (maximum a posteriori) estimate
    map_result = np.argmax(Q, axis=0).reshape((height, width))
    
    return map_result


def map_labels_back(crf_output, label_mapping):
    """
    Map CRF integer labels back to original mask values (0, 128, 255)
    
    Parameters:
    -----------
    crf_output : numpy array
        CRF output with integer labels (0, 1, 2)
    label_mapping : dict
        Mapping from integer to original values
    
    Returns:
    --------
    final_mask : numpy array
        Mask with original label values
    """
    final_mask = np.zeros_like(crf_output, dtype=np.uint8)
    for int_label, orig_value in label_mapping.items():
        final_mask[crf_output == int_label] = orig_value
    return final_mask


# ============================================================================
# PREDICTION WITH CRF
# ============================================================================

def predict_with_crf(clf, scaler, test_image, use_both_colorspaces=True,
                     apply_crf=True, crf_params=None):
    """
    Predict segmentation with optional CRF refinement
    
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
    apply_crf : bool
        Whether to apply CRF refinement
    crf_params : dict or None
        CRF parameters
    
    Returns:
    --------
    final_mask : numpy array
        Segmentation mask (with original label values: 0, 128, 255)
    naive_bayes_mask : numpy array
        Raw Naive Bayes prediction (before CRF)
    """
    # Get probability map from Naive Bayes
    probabilities = get_probability_map(clf, scaler, test_image, use_both_colorspaces)
    
    # Get class labels (to map between integer indices and original values)
    class_labels = clf.classes_  # e.g., [0, 128, 255]
    label_to_int = {label: i for i, label in enumerate(class_labels)}
    int_to_label = {i: label for i, label in enumerate(class_labels)}
    
    # Naive Bayes prediction (before CRF)
    naive_prediction = np.argmax(probabilities, axis=0)
    naive_bayes_mask = map_labels_back(naive_prediction, int_to_label)
    
    if not apply_crf:
        return naive_bayes_mask, naive_bayes_mask
    
    # Apply CRF refinement
    print("Applying Dense CRF refinement...")
    
    # Prepare image in RGB format
    image_rgb = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    
    # Set default CRF parameters if not provided
    if crf_params is None:
        crf_params = {
            'n_iters': 5,
            'sxy_gaussian': 3,
            'compat_gaussian': 3,
            'sxy_bilateral': 80,
            'srgb_bilateral': 13,
            'compat_bilateral': 10
        }
    
    # Apply CRF
    crf_output = apply_dense_crf(image_rgb, probabilities, **crf_params)
    
    # Map back to original label values
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
    from sklearn.metrics import accuracy_score
    
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
        axes[2].set_title('Naive Bayes (Before CRF)')
        axes[2].axis('off')
        
        axes[3].imshow(crf_mask, cmap='gray')
        axes[3].set_title('After CRF Refinement')
        axes[3].axis('off')
    else:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(naive_mask, cmap='gray')
        axes[1].set_title('Naive Bayes (Before CRF)')
        axes[1].axis('off')
        
        axes[2].imshow(crf_mask, cmap='gray')
        axes[2].set_title('After CRF Refinement')
        axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main execution pipeline with CRF"""
    
    # Configuration
    TRAIN_FUNDUS_DIR = 'REFUGE2\\Train\\REFUGE1-train\\Training400\\Glaucoma'
    TRAIN_MASK_DIR = 'REFUGE2\\Train\\REFUGE1-train\\Disc_Cup_Masks\\Glaucoma'
    TEST_FUNDUS_DIR = 'REFUGE2\\Test\\refuge2-test'
    TEST_MASK_DIR = 'REFUGE2\\Test\\Disc_Mask'  # Optional, for evaluation
    OUTPUT_DIR = 'NaiveBayesClassifierOutput'
    
    USE_BOTH_COLORSPACES = True
    APPLY_CRF = True
    
    # CRF parameters (tune these for best results)
    CRF_PARAMS = {
        'n_iters': 10,              # More iterations = smoother results
        'sxy_gaussian': 3,          # Spatial smoothness
        'compat_gaussian': 3,       # Gaussian compatibility
        'sxy_bilateral': 80,        # Bilateral spatial (larger = more spatial smoothing)
        'srgb_bilateral': 13,       # Bilateral color (smaller = only similar colors grouped)
        'compat_bilateral': 10      # Bilateral compatibility
    }
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # ========== TRAINING ==========
    print("="*60)
    print("TRAINING PHASE")
    print("="*60)
    
    fundus_files = sorted(glob(os.path.join(TRAIN_FUNDUS_DIR, '*.*')))
    mask_files = sorted(glob(os.path.join(TRAIN_MASK_DIR, '*.*')))
    
    fundus_train = [cv2.imread(f) for f in fundus_files]
    masks_train = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in mask_files]
    
    clf, scaler = train_naive_bayes(fundus_train, masks_train, 
                                    use_both_colorspaces=USE_BOTH_COLORSPACES)
    
    # ========== TESTING ==========
    print("\n" + "="*60)
    print("TESTING PHASE WITH CRF REFINEMENT")
    print("="*60)
    
    test_files = sorted(glob(os.path.join(TEST_FUNDUS_DIR, '*.*')))
    
    metrics_nb = []   # Naive Bayes metrics
    metrics_crf = []  # CRF metrics
    
    for i, test_file in enumerate(test_files):
        print(f"\nProcessing test image {i+1}/{len(test_files)}: {os.path.basename(test_file)}")
        
        test_image = cv2.imread(test_file)
        if test_image is None:
            continue
        
        # Predict with CRF
        crf_mask, nb_mask = predict_with_crf(
            clf, scaler, test_image,
            use_both_colorspaces=USE_BOTH_COLORSPACES,
            apply_crf=APPLY_CRF,
            crf_params=CRF_PARAMS
        )
        
        # Evaluate if ground truth exists
        test_basename = os.path.basename(test_file)
        gt_file = os.path.join(TEST_MASK_DIR, test_basename)
        
        if os.path.exists(gt_file):
            gt_mask = cv2.imread(gt_file, cv2.IMREAD_GRAYSCALE)
            
            # Evaluate both methods
            metrics_nb_img = evaluate_segmentation(nb_mask, gt_mask)
            metrics_crf_img = evaluate_segmentation(crf_mask, gt_mask)
            
            metrics_nb.append(metrics_nb_img)
            metrics_crf.append(metrics_crf_img)
            
            print(f"\nNaive Bayes:")
            print(f"  Accuracy: {metrics_nb_img['accuracy']:.4f}")
            print(f"  Mean Dice: {metrics_nb_img['dice_mean']:.4f}")
            
            print(f"\nWith CRF:")
            print(f"  Accuracy: {metrics_crf_img['accuracy']:.4f}")
            print(f"  Mean Dice: {metrics_crf_img['dice_mean']:.4f}")
            print(f"  Improvement: {(metrics_crf_img['dice_mean'] - metrics_nb_img['dice_mean']):.4f}")
        else:
            gt_mask = None
        
        # Save results
        base_name = os.path.splitext(test_basename)[0]
        
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base_name}_nb_mask.png"), nb_mask)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base_name}_crf_mask.png"), crf_mask)
        
        # Visualize comparison
        viz_path = os.path.join(OUTPUT_DIR, f"{base_name}_comparison.png")
        visualize_crf_comparison(test_image, nb_mask, crf_mask, gt_mask, viz_path)
    
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
        print(f"\nOverall Dice Improvement: {improvement:.4f}")


if __name__ == "__main__":
    main()
