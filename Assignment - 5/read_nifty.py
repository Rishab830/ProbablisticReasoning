import SimpleITK as sitk
import matplotlib.pyplot as plt

paths = ["mp_cyra4454.1999-08-23.mri_at2.nii", "mp_cyra4454.1999-08-23.mri_at1_t2space.nii", "mp_cyra4454.1999-08-23.classified_nat2.nii"]
imgs = []

for path in paths:
    img = sitk.ReadImage(path)
    imgs.append(img)

datas = []
for img in imgs:
    data = sitk.GetArrayFromImage(img)  # z, y, x order
    datas.append(data)

# Show a slice
for data in datas:
    plt.imshow(data[data.shape[0] // 2], cmap='gray')
    plt.title('Middle Slice')
    plt.show()