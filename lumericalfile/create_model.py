import sys
import os
import numpy as np

# 1. 環境路徑設定
api_path = r"C:\Program Files\Lumerical\v261\api\python"
bin_path = r"C:\Program Files\Lumerical\v261\bin"

if api_path not in sys.path:
    sys.path.append(api_path)

if os.path.exists(bin_path):
    os.add_dll_directory(bin_path)

import lumapi

# 2. 啟動 FDTD
print("🚀 正在啟動 Lumerical FDTD...")
fdtd = lumapi.FDTD()

# 3. 定義核心物理參數
wavelength = 0.532e-6  # 532 nm
pitch = 0.4e-6         # 400 nm
height = 0.6e-6        # 600 nm
r_init = 0.1e-6        # 初始半徑

# 4. 建立幾何結構
fdtd.addrect()
fdtd.set("name", "Substrate")
fdtd.setnamed("Substrate", "x span", pitch)
fdtd.setnamed("Substrate", "y span", pitch)
fdtd.setnamed("Substrate", "z min", -0.5e-6)
fdtd.setnamed("Substrate", "z max", 0)
fdtd.setnamed("Substrate", "material", "SiO2 (Glass) - Palik")

fdtd.addcircle()
fdtd.set("name", "NanoPillar")
fdtd.setnamed("NanoPillar", "x", 0)
fdtd.setnamed("NanoPillar", "y", 0)
fdtd.setnamed("NanoPillar", "z min", 0)
fdtd.setnamed("NanoPillar", "z max", height)
fdtd.setnamed("NanoPillar", "radius", r_init)
fdtd.setnamed("NanoPillar", "material", "Si3N4 (Silicon Nitride) - Luke")
# 5. 設定模擬區域 (FDTD Region)
fdtd.addfdtd()
fdtd.setnamed("FDTD", "dimension", "3D")
fdtd.setnamed("FDTD", "x span", pitch)
fdtd.setnamed("FDTD", "y span", pitch)
fdtd.setnamed("FDTD", "z span", 2.0e-6)
fdtd.setnamed("FDTD", "z", height / 2)

fdtd.setnamed("FDTD", "x min bc", "Periodic")
fdtd.setnamed("FDTD", "y min bc", "Periodic")
fdtd.setnamed("FDTD", "z min bc", "PML")
fdtd.setnamed("FDTD", "z max bc", "PML")

fdtd.setnamed("FDTD", "mesh accuracy", 2)
fdtd.setnamed("FDTD", "force symmetric x mesh", True)
fdtd.setnamed("FDTD", "force symmetric y mesh", True)

# 6. 加入光源
fdtd.addplane()
fdtd.setnamed("source", "name", "Source")
fdtd.setnamed("Source", "injection axis", "z-axis")
fdtd.setnamed("Source", "direction", "Forward")
fdtd.setnamed("Source", "x span", pitch)
fdtd.setnamed("Source", "y span", pitch)
fdtd.setnamed("Source", "z", -0.2e-6)
fdtd.setnamed("Source", "center wavelength", wavelength)
fdtd.setnamed("Source", "wavelength span", 0)

# 7. 設定監測器 (Monitor)
fdtd.adddftmonitor()
fdtd.setnamed("monitor", "name", "Transmission_Monitor")
fdtd.setnamed("Transmission_Monitor", "monitor type", "2D Z-normal")
fdtd.setnamed("Transmission_Monitor", "x span", pitch)
fdtd.setnamed("Transmission_Monitor", "y span", pitch)
fdtd.setnamed("Transmission_Monitor", "z", height + 0.3e-6) 

# --- 根據你的 Warning 訊息進行的優化更新 ---
fdtd.setnamed("Transmission_Monitor", "override global monitor settings", True)
# 使用新版參數：use wavelength spacing = 1 (Linear)
fdtd.setnamed("Transmission_Monitor", "use wavelength spacing", 1) 
fdtd.setnamed("Transmission_Monitor", "frequency points", 1)

# 8. 保存
filename = "Metasurface_UnitCell_V261_Compatible_Si3N4.fsp"
fdtd.save(filename)

print("\n" + "="*40)
print(f"✅ 模型全部修正完成且符合新版 API！")
print(f"檔案已保存為: {filename}")
print("="*40)

input("\n👉 [按回車鍵] 將關閉 FDTD 並結束程式...")