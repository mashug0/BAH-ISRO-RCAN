import pandas as pd
import matplotlib.pyplot as plt
# from Degrador import compute_ssim

df = pd.read_csv('Data/MOS.csv')
plt.hist(df['mos_score'], bins=40, edgecolor='black')  # Change bins as needed
plt.title('Score Distribution')
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
