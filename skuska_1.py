import torch
import numpy as np
import matplotlib.pyplot as plt

data = torch.load(r"E:\Features_For_BT\00155_hr_decomposition_lvl_4.pt")

print(type(data))

if isinstance(data, torch.Tensor):
    print("Shape:", data.shape)
    print("Dtype:", data.dtype)
    print("Min:", data.min())
    print("Max:", data.max())

import matplotlib.pyplot as plt

# odstránime prvý rozmer [1, 48, 1250] -> [48, 1250]
data_2d = data.squeeze(0)

print("New shape:", data_2d.shape)

plt.figure(figsize=(12, 6))

plt.imshow(
    data_2d.numpy(),
    aspect="auto"
)

plt.xlabel("Position")
plt.ylabel("Feature channel")
plt.title("CNN feature map")

plt.colorbar(label="Value")
plt.show()