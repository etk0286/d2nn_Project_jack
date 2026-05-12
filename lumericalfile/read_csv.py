import pandas as pd
import matplotlib.pyplot as plt

# 1. 讀取 CSV 檔案
df = pd.read_csv('Metasurface_Lookup_Table.csv')

# 2. 查看前幾行數據（確認讀取成功）
print(df.head())

# 3. 畫圖
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# 畫相位
ax1.plot(df['Radius_nm'], df['Phase_rad'], 'b-o', label='Phase')
ax1.set_ylabel('Phase (rad)')
ax1.grid(True, linestyle='--')
ax1.legend()

# 畫穿透率
ax2.plot(df['Radius_nm'], df['Transmission'], 'r-s', label='Transmission')
ax2.set_xlabel('Radius (nm)')
ax2.set_ylabel('Efficiency (T)')
ax2.set_ylim(0, 1.1)
ax2.grid(True, linestyle='--')
ax2.legend()

plt.tight_layout()
plt.show()