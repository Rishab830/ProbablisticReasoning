
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from sklearn.mixture import GaussianMixture

paths = ["mp_cyra4454.1999-08-23.mri_at2.nii", 
         "mp_cyra4454.1999-08-23.mri_at1_t2space.nii", 
         "mp_cyra4454.1999-08-23.classified_nat2.nii"]
imgs = []

# Load images
for path in paths:
    img = sitk.ReadImage(path)
    imgs.append(img)

# Convert to numpy arrays
datas = []
for img in imgs:
    data = sitk.GetArrayFromImage(img)  # z, y, x order
    datas.append(data)

# Fit Gaussian distributions to each tissue class for each file
for idx, (data, path) in enumerate(zip(datas, paths)):
    filename = path.split('/')[-1]

    print("="*70)
    print(f"Processing: {filename}")
    print("="*70)

    # Get non-zero intensities
    intensities = data.flatten()
    intensities_nonzero = intensities[intensities > 0]

    # Fit Gaussian Mixture Model with 4 components (CSF, GM, WM, Lesions)
    print("Fitting 4 Gaussian components...")
    intensities_reshaped = intensities_nonzero.reshape(-1, 1)
    gmm = GaussianMixture(n_components=4, random_state=42, max_iter=200)
    gmm.fit(intensities_reshaped)

    # Sort components by mean (typically: CSF < GM < WM < Lesions)
    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_.flatten())
    weights = gmm.weights_

    sorted_indices = np.argsort(means)
    tissue_labels = ['CSF', 'Grey Matter', 'White Matter', 'Lesions']

    # Store sorted parameters
    components = []
    for i, idx_sort in enumerate(sorted_indices):
        component = {
            'label': tissue_labels[i],
            'mean': means[idx_sort],
            'std': stds[idx_sort],
            'weight': weights[idx_sort]
        }
        components.append(component)
        print(f"{component['label']:15s}: mean={component['mean']:7.2f}, "
              f"std={component['std']:6.2f}, weight={component['weight']:.3f}")

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot middle slice
    ax1.imshow(data[data.shape[0] // 2], cmap='gray')
    ax1.set_title(f'Middle Slice - {filename}', fontsize=12, fontweight='bold')
    ax1.axis('off')

    # Plot histogram with fitted Gaussians
    hist, bin_edges = np.histogram(intensities_nonzero, bins=256, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Plot histogram
    ax2.bar(bin_centers, hist, width=bin_edges[1]-bin_edges[0], 
            alpha=0.4, color='gray', label='Intensity Histogram', edgecolor='black')

    # Plot individual Gaussian components
    x_range = np.linspace(intensities_nonzero.min(), intensities_nonzero.max(), 1000)
    colors = ['blue', 'green', 'red', 'orange']

    for i, comp in enumerate(components):
        gaussian_curve = comp['weight'] * norm.pdf(x_range, comp['mean'], comp['std'])
        ax2.plot(x_range, gaussian_curve, colors[i], linewidth=2.5, 
                label=f"{comp['label']} (μ={comp['mean']:.1f}, σ={comp['std']:.1f})")

    # Plot combined mixture
    mixture_pdf = np.zeros_like(x_range)
    for comp in components:
        mixture_pdf += comp['weight'] * norm.pdf(x_range, comp['mean'], comp['std'])

    ax2.plot(x_range, mixture_pdf, 'k--', linewidth=2.5, label='Mixture Model')

    ax2.set_xlabel('Intensity', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Probability Density', fontsize=12, fontweight='bold')
    ax2.set_title(f'Gaussian Mixture Model - {filename}', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9, loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_name = f'gaussian_fit_{filename.replace(".nii", "")}.png'
    plt.savefig(output_name, dpi=300, bbox_inches='tight')
    print(f"✓ Saved as: {output_name}\n")
    plt.show()

print("="*70)
print("All Gaussian fittings completed!")
print("="*70)
