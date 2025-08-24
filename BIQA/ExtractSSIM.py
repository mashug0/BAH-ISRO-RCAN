from Degrador import compute_ssim
import pandas as pd
def add_ssim(ref_dir , degrade_dir , csv):
    df = pd.read_csv(csv)
    filenames = df['filename']
    