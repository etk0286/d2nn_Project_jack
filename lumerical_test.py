import sys, os

# 1. 設定 Lumerical 環境路徑 (使用你原本的 v261 路徑)
api_path = r"C:\Program Files\Lumerical\v261\api\python"
bin_path = r"C:\Program Files\Lumerical\v261\bin"
if api_path not in sys.path: sys.path.append(api_path)
if os.path.exists(bin_path): os.add_dll_directory(bin_path)

import lumapi

print("步驟 1: 正在嘗試透過 Python 開啟 FDTD...")
try:
    # 這裡故意設定 hide=False，讓你親眼看到軟體被 Python 打開
    fdtd = lumapi.FDTD(hide=False) 
    print("✅ FDTD 開啟成功！License 正常。")

    print("步驟 2: 正在嘗試建立測試物件...")
    fdtd.addrect()
    fdtd.set("name", "Hello_D2NN")
    print("✅ 物件建立成功！")

    print("步驟 3: 正在嘗試存檔與關閉...")
    # 存一個測試檔，確認沒有 File Lock 問題
    fdtd.save("API_Connection_Test.fsp") 
    fdtd.close()
    print("✅ 存檔並成功關閉！背景已釋放。")
    
    print("\n🎉 恭喜！你的 Lumerical 和 Python 連線非常完美，可以繼續跑 D2NN 模擬了！")

except Exception as e:
    print(f"\n❌ 發生錯誤，代表環境還有問題：\n{e}")