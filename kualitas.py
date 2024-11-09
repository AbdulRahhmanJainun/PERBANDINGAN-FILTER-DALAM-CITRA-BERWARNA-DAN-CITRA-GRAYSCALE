import cv2
import numpy as np
import matplotlib.pyplot as plt

# Fungsi untuk menerapkan low-pass filter (blur)
def low_pass_filter(image, kernel_size=5):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

# Fungsi untuk menerapkan high-pass filter
def high_pass_filter(image, kernel_size=5):
    low_pass = low_pass_filter(image, kernel_size)
    return cv2.subtract(image, low_pass)

# Fungsi untuk menerapkan high-boost filter
def high_boost_filter(image, kernel_size=5, boost_factor=1.5):
    high_pass = high_pass_filter(image, kernel_size)
    return cv2.addWeighted(image, boost_factor, high_pass, 1 - boost_factor, 0)

# Membaca citra berwarna
image_color = cv2.imread ("C:\\gambar\\istockphoto-1317323736-612x612.jpg")
image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)

# Terapkan filter pada citra grayscale
low_pass_gray = low_pass_filter(image_gray)
high_pass_gray = high_pass_filter(image_gray)
high_boost_gray = high_boost_filter(image_gray)

# Terapkan filter pada citra berwarna (per saluran RGB)
low_pass_color = cv2.merge([low_pass_filter(image_color[:,:,i]) for i in range(3)])
high_pass_color = cv2.merge([high_pass_filter(image_color[:,:,i]) for i in range(3)])
high_boost_color = cv2.merge([high_boost_filter(image_color[:,:,i]) for i in range(3)])

# Menampilkan hasil filter
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# Citra Grayscale
axes[0, 0].imshow(image_gray, cmap='gray')
axes[0, 0].set_title("Original Grayscale")
axes[0, 1].imshow(low_pass_gray, cmap='gray')
axes[0, 1].set_title("Low-Pass Grayscale")
axes[0, 2].imshow(high_pass_gray, cmap='gray')
axes[0, 2].set_title("High-Pass Grayscale")
axes[0, 3].imshow(high_boost_gray, cmap='gray')
axes[0, 3].set_title("High-Boost Grayscale")

# Citra Berwarna
axes[1, 0].imshow(cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title("Original Color")
axes[1, 1].imshow(cv2.cvtColor(low_pass_color, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title("Low-Pass Color")
axes[1, 2].imshow(cv2.cvtColor(high_pass_color, cv2.COLOR_BGR2RGB))
axes[1, 2].set_title("High-Pass Color")
axes[1, 3].imshow(cv2.cvtColor(high_boost_color, cv2.COLOR_BGR2RGB))
axes[1, 3].set_title("High-Boost Color")

# Menghapus axis dan menampilkan hasil
for ax in axes.flat:
    ax.axis('off')
plt.tight_layout()
plt.show()
