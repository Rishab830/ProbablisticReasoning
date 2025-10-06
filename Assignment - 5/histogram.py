import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np

paths = ["mp_cyra4454.1999-08-23.mri_at2.nii", 
         "mp_cyra4454.1999-08-23.mri_at1_t2space.nii", 
         "mp_cyra4454.1999-08-23.classified_nat2.nii"]
imgs = []

for path in paths:
    img = sitk.ReadImage(path)
    imgs.append(img)

datas = []
for img in imgs:
    data = sitk.GetArrayFromImage(img)
    datas.append(data)

# Show a slice and histogram for each file
for idx, (data, path) in enumerate(zip(datas, paths)):
    filename = path.split('/')[-1]
    
    # Create figure with 2 subplots side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot middle slice
    ax1.imshow(data[data.shape[0] // 2], cmap='gray')
    ax1.set_title(f'Middle Slice - {filename}')
    ax1.axis('off')
    
    # Plot histogram (remove zeros for better visualization)
    intensities = data.flatten()
    intensities_nonzero = intensities[intensities > 0]
    
    ax2.hist(intensities_nonzero, bins=100, color='steelblue', 
             edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Intensity', fontweight='bold')
    ax2.set_ylabel('Frequency', fontweight='bold')
    ax2.set_title(f'Histogram - {filename}')
    ax2.grid(True, alpha=0.3)
    
    # Add statistics
    stats = f'Mean: {intensities_nonzero.mean():.1f}\nStd: {intensities_nonzero.std():.1f}'
    ax2.text(0.98, 0.98, stats, transform=ax2.transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()
