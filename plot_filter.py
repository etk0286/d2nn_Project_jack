import matplotlib.pyplot as plt
import numpy as np
import os

# 使用 r 前綴避免路徑轉義錯誤
file_path = r'C:\Users\etk02\Desktop\光學比賽\d2nn_example\Diffractive-Deep-Neural-Networks\filter_height_map.npy'

if os.path.exists(file_path):
    filter_map = np.load(file_path)
    
    # 檢查數據維度
    print(f"數據讀取成功！矩陣大小為: {filter_map.shape}")
    
    # 繪製圖像
    plt.figure(figsize=(6, 6))
    plt.imshow(filter_map, cmap='viridis') # 使用 viridis 顏色比較好觀察厚度差
    plt.colorbar(label='Height / Intensity')
    plt.title("Visualization of Filter Layer (Input Image)")
    plt.show()
else:
    print(f"錯誤：找不到檔案，請檢查路徑是否正確：\n{file_path}")