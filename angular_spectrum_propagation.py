import numpy as np
import matplotlib.pyplot as plt

def angular_spectrum_propagation(U_in, wavelength, dx, z):
    """
    使用角譜法計算光場的自由空間傳播
    U_in: 輸入光場 (2D numpy 複數陣列)
    wavelength: 光波長 (m)
    dx: 像素尺寸 (m)
    z: 傳播距離 (m)
    """
    N = U_in.shape[0] # 假設是 NxN 的正方形陣列
    
    # 1. 建立空間頻率網格 (Spatial Frequency Grid)
    fx = np.fft.fftfreq(N, d=dx)
    fy = np.fft.fftfreq(N, d=dx)
    Fx, Fy = np.meshgrid(fx, fy)
    
    # 2. 計算傳播核 H (Transfer Function)
    # 計算根號內部項目
    term = 1 - (wavelength * Fx)**2 - (wavelength * Fy)**2
    
    # 濾除漸逝波 (Evanescent waves)，防止數值發散
    term[term < 0] = 0 
    
    # 產生相位延遲矩陣 H
    H = np.exp(1j * 2 * np.pi * (z / wavelength) * np.sqrt(term))
    
    # 3. 角譜法核心計算： FFT -> 乘 H -> IFFT
    U_fft = np.fft.fft2(U_in)          # 轉到角譜域
    U_z_fft = U_fft * H                # 空間傳播
    U_z = np.fft.ifft2(U_z_fft)        # 轉回空間域
    
    return U_z

# ================= 測試與驗證區塊 =================

# 設定物理參數
N = 1024               # 網格大小 (1024 x 1024)
L = 1e-3               # 物理視窗大小 1 mm (1e-3 m)
dx = L / N             # 每個像素的物理尺寸
wavelength = 632.8e-9  # 光波長 (例如 He-Ne 雷射 632.8 nm)
z = 7e-3               # 傳播距離 5 mm

# 建立空間座標網格 (為了畫輸入圖案)
x = np.linspace(-L/2, L/2, N)
y = np.linspace(-L/2, L/2, N)
X, Y = np.meshgrid(x, y)

# 定義輸入光場 U0：一個半徑為 50 微米的圓孔透光區
U0 = np.zeros((N, N), dtype=complex)
radius = 50e-6
U0[X**2 + Y**2 <= radius**2] = 1.0

# 執行角譜傳播
U_z = angular_spectrum_propagation(U0, wavelength, dx, z)

# 計算光強 (Intensity = 振幅的平方)
I_0 = np.abs(U0)**2
I_z = np.abs(U_z)**2

# 繪圖驗證
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.title("Input Plane ($z=0$)\nCircular Aperture")
plt.imshow(I_0, cmap='gray', extent=[-L/2*1e3, L/2*1e3, -L/2*1e3, L/2*1e3])
plt.xlabel("x (mm)")
plt.ylabel("y (mm)")

plt.subplot(1, 2, 2)
plt.title(f"Output Plane ($z={z*1e3}$ mm)\nDiffraction Pattern")
# 為了讓繞射外圈更明顯，我們把亮度做 Gamma 校正 (開根號) 取代直接顯示
plt.imshow(np.sqrt(I_z), cmap='magma', extent=[-L/2*1e3, L/2*1e3, -L/2*1e3, L/2*1e3])
plt.xlabel("x (mm)")
plt.ylabel("y (mm)")
plt.colorbar(label="Amplitude (sqrt of Intensity)")

plt.subplots_adjust(wspace=0.1)
plt.show()