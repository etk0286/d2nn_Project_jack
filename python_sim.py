import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ==========================================
# 1. 建立離散像素網格 (感測器平面)
# ==========================================
# 假設我們有一個 64x64 像素的偵測器陣列
grid_size = 64
x_max = 12.8
y_min = -12.8

# 初始化空白感測器
sensor_matrix = np.zeros((grid_size, grid_size))

# 建立一個輔助函數：將物理座標 (x, y) 轉換為矩陣的 (row, col) 索引
def coord_to_index(x, y):
    col = int((x / x_max) * (grid_size - 1))
    # y 軸從 -12.8 對應到 row 0，0 對應到 row 63
    row = int(((y - y_min) / -y_min) * (grid_size - 1))
    return row, col

# 建立一個輔助函數：在感測器上點亮一個 n x n 像素的「離散光點」
def add_discrete_spot(matrix, x, y, intensity, size=1):
    row, col = coord_to_index(x, y)
    r_start = max(0, row - size)
    r_end = min(grid_size, row + size + 1)
    c_start = max(0, col - size)
    c_end = min(grid_size, col + size + 1)
    matrix[r_start:r_end, c_start:c_end] += intensity

# ==========================================
# 2. 寫入第一張圖的目標特徵 (改為方形像素點)
# ==========================================
# 依照您第一張圖的位置，打上強弱不同的光點
# 右上方最強的主光點
add_discrete_spot(sensor_matrix, x=9.0, y=-0.5, intensity=2.5, size=1)
# 其下方的次強光點
add_discrete_spot(sensor_matrix, x=9.0, y=-2.5, intensity=1.8, size=1)
# 右側邊緣的中等光點
add_discrete_spot(sensor_matrix, x=10.5, y=-9.0, intensity=1.0, size=0) 
# 底部偏左的微弱光點
add_discrete_spot(sensor_matrix, x=6.0, y=-12.0, intensity=0.7, size=0)

# ==========================================
# 3. 疊加感測器雜訊 (Sensor Noise)
# ==========================================
# 加入均勻的隨機背景雜訊，模擬 CMOS 的暗電流與讀取雜訊
noise = np.random.normal(loc=0.3, scale=0.2, size=(grid_size, grid_size))
final_image = sensor_matrix + noise

# 確保亮度值不會小於 0
final_image = np.clip(final_image, 0, None)

# ==========================================
# 4. 繪圖 (呈現顆粒感與紅框)
# ==========================================
fig, ax = plt.subplots(figsize=(8, 6))

# 【關鍵設定】： interpolation='nearest' 會讓圖形顯示出銳利的像素方塊
# cmap='viridis' 對應您第二張圖的紫-黃漸層配色
im = ax.imshow(final_image, origin='lower', extent=[0, x_max, y_min, 0], 
               cmap='viridis', interpolation='nearest')

ax.set_xlabel('x(m) (x10^-3)', fontsize=12)
ax.set_ylabel('y(m) (x10^-3)', fontsize=12)
ax.set_title("Discrete Light Spots Distribution (Sensor View)")

# 加上 Colorbar
fig.colorbar(im, ax=ax, label='Signal Intensity')



plt.tight_layout()
plt.show()