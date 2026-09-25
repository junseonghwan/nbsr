import subprocess
import numpy as np

x = np.array([0, 20, 40, 60, 80, 100])
for i in range(len(x)):
    for j in range(i+1, len(x)):
        data_path = f"/Users/sjun6/Dropbox/seong/miRNA/data/Ernesto/{x[i]}S_{x[j]}S/"
        print(data_path)
        subprocess.run(["python",
                        "/Users/sjun6/Dropbox/seong/nbsr/nbsr/main.py",
                        "results",
                        data_path,
                        "mixture_group",
                        f"{x[j]}S",
                        f"{x[i]}S"])

