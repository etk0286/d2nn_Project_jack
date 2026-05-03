import sys, os
import numpy as np
import matplotlib.pyplot as plt

# --- 1. 設定 Lumerical 環境路徑 ---
api_path = r"C:\Program Files\Lumerical\v261\api\python"
bin_path = r"C:\Program Files\Lumerical\v261\bin"
if api_path not in sys.path: sys.path.append(api_path)
if os.path.exists(bin_path): os.add_dll_directory(bin_path)

import lumapi

# --- 2. 參數與檔案名稱設定 ---
file_name = r"C:\Users\etk02\Desktop\optical_race\d2nn_example\D2NN_Simulation.fsp"   # 剛剛成功合併的主檔案
monitor_name = "detector_plane_monitor"    # 探測器監視器的名稱

print("🚀 正在背景啟動 Lumerical FDTD 引擎 (不開啟介面)...")
# hide=True 是節省 RAM 和顯卡資源的關鍵
fdtd = lumapi.FDTD(hide=True)

try:
    print(f"📂 正在載入檔案: {file_name}")
    fdtd.load(file_name)

    print("⏳ 開始執行 FDTD 模擬計算... (這可能需要數分鐘到數小時，請耐心等候)")
    # 這行指令會讓 Lumerical 開始跑進度條，直到模擬 100% 結束才會繼續往下執行
    fdtd.run()
    print("✅ 模擬計算完成！")

    print("📊 正在從探測器提取數據...")
    # 提取 X 和 Y 座標 (轉換成一維陣列)
    x = fdtd.getdata(monitor_name, "x").flatten()
    y = fdtd.getdata(monitor_name, "y").flatten()
    
    # 提取電場強度的平方 (|E|^2)，這最能代表光斑的能量分佈
    E2 = fdtd.getelectric(monitor_name)
    
    # E2 從 Lumerical 吐出來的維度通常是 (x, y, z, f)，我們用 squeeze 把它壓平成二維矩陣 (x, y)
    intensity = np.squeeze(E2)

    print("🧹 數據提取完畢，立刻關閉 FDTD 以釋放所有 RAM！")
    fdtd.close()
    
    # --- 3. 開始使用 Matplotlib 繪圖 ---
    print("🎨 正在生成能量分佈圖...")
    plt.figure(figsize=(8, 6))
    
    # 建立網格，並將座標單位從公尺轉換為微米 (um)，方便閱讀
    X, Y = np.meshgrid(x * 1e6, y * 1e6) 
    
    # 注意：Matplotlib 畫圖時的矩陣方向與 Lumerical 預設的維度相反，所以 intensity 需要做轉置 (.T)
    im = plt.pcolormesh(X, Y, intensity.T, cmap='jet', shading='auto')
    
    plt.colorbar(im, label='Electric Field Intensity $|E|^2$')
    plt.title('D2NN Detector Plane Energy Distribution')
    plt.xlabel('X position ($\mu m$)')
    plt.ylabel('Y position ($\mu m$)')
    
    # 讓 X 軸與 Y 軸比例一比一，避免光斑變形
    plt.axis('equal')
    plt.tight_layout()
    
    # 儲存圖片並顯示
    output_img = "D2NN_Result_Map.png"
    plt.savefig(output_img, dpi=300)
    print(f"📸 圖片已高畫質儲存至當前目錄：{output_img}")
    
    plt.show()

except Exception as e:
    print(f"\n❌ 執行過程中發生錯誤：\n{e}")
    # 確保發生錯誤時，也會把背景的 Lumerical 關掉，保護 License 和記憶體
    fdtd.close()