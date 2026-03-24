import scipy.io
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

mat = scipy.io.loadmat(PROJECT_ROOT / "data/processed/train_sim/mat/eit_simulation_data.mat")
target = mat['target_data']
# 取中间一帧
img = target[0, 25, :, :]

plt.figure(figsize=(5,5))
plt.imshow(img, cmap='bone')
plt.title("Check Ground Truth")
plt.colorbar()
output_path = PROJECT_ROOT / "outputs/figures/check_ground_truth.png"
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path)
plt.show()
