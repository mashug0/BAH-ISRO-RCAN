import os
import csv

def save_labels(in_dir):

    csv_path = os.path.join(in_dir , "MOS.csv")
    with open(csv_path, "w", newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["filename", "score"])

        for root, _, filenames in os.walk(in_dir):
            for filename in filenames:
                if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
                    continue

                base_name = os.path.splitext(filename)[0]

                try:
                    mos = float(base_name[-6:])
                    mos_str = f"{mos:.4f}"
                    writer.writerow([base_name , mos_str])
                except Exception as e:
                    print(f"Error reading {base_name}")
                    continue
        print("Writing Complete")

if __name__ == "__main__":
    save_labels("Data/images")