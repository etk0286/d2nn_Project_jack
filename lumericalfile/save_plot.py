import sys
import os
import numpy as np
import matplotlib.pyplot as plt

import datetime
# 1. 環境路徑設定
api_path = r"C:\Program Files\Lumerical\v261\api\python"
bin_path = r"C:\Program Files\Lumerical\v261\bin"
if api_path not in sys.path: sys.path.append(api_path)
if os.path.exists(bin_path): os.add_dll_directory(bin_path)

import lumapi

# 2. 檔案路徑
file_path = r"C:\Users\etk02\Desktop\光學比賽\Metasurface_UnitCell_V261_Compatible_TI02.fsp"


if not os.path.exists(file_path):
    print(f"❌ 找不到檔案：{file_path}")
    sys.exit()

# 3. 啟動 FDTD 並載入
print(f"🚀 正在讀取檔案並提取數據...")
fdtd = lumapi.FDTD()
fdtd.load(file_path)

# 請確認你的 Sweep 名稱
sweep_name = "sweep" 

try:
    print(f"🏃 正在執行 Sweep: {sweep_name}...")
    fdtd.runsweep(sweep_name)
    
    # 1. 提取結果
    E_results = fdtd.getsweepresult(sweep_name, "E")
    T_results = fdtd.getsweepresult(sweep_name, "T")

    # 2. 提取半徑 R 與穿透率 T
    radius = E_results["R"].flatten()
    transmission = T_results["T"].flatten() # 提取傳輸率數據

    # 3. 從 'E' 數據集中提取電場分量
    E_full = E_results["E"]
    
    # 取得維度資訊
    dims = E_full.shape
    print(f"📊 電場矩陣維度: {dims}") 
    
    # 4. 提取中心點的 Ex
    mid_x, mid_y = dims[0] // 2, dims[1] // 2
    Ex_center = E_full[mid_x, mid_y, 0, 0, :, 0] 

    # 5. 相位計算與 Unwrap
    phase = np.unwrap(np.angle(Ex_center))

    # --- 6. 繪圖部分 (修改為上下兩張圖) ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # 上圖：相位圖 (Phase)
    ax1.plot(radius * 1e9, phase, 'b-o', label='Phase (rad)')
    ax1.set_ylabel('Phase (rad)', fontsize=12)
    ax1.set_title(f'Metasurface Analysis: {os.path.basename(file_path)}', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()

    # 下圖：穿透率圖 (Transmission)
    ax2.plot(radius * 1e9, transmission, 'r-s', label='Transmission (T)')
    ax2.set_xlabel('Radius (nm)', fontsize=12)
    ax2.set_ylabel('Efficiency (T)', fontsize=12)
    ax2.set_ylim(0, 1.1) # 設定 T 的範圍在 0 到 1.1 之間
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()

    plt.tight_layout()
    plt.show()
    
    print("✅ 數據提取與繪圖成功完成！")

except Exception as e:
    print(f"❌ 錯誤詳細資訊：{e}")
    import traceback
    traceback.print_exc()

data_to_save = np.column_stack((radius * 1e9, phase, transmission))

# 儲存為 CSV 檔案
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
filename = f"Metasurface_LUT_{timestamp}.csv"
header = "Radius_nm,Phase_rad,Transmission"
np.savetxt(filename, data_to_save, delimiter=",", header=header, comments='')