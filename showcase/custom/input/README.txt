把你自己的图放在本目录即可（jpg / png，命名随意，数量不限）。

注意：只需要"原图"—— 不需要 mask (B)，也不需要真值 GT (C)。
脚本会为 RASM 自动估一个粗阴影 mask 来驱动模型（仅用于推理，保存为 outputs/_masks/，不是真值）。

然后回到 RASM/ 根目录执行：
    bash showcase/custom/run_custom.sh
