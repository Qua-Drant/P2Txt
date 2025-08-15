import numpy as np

# 用软件修改后保存的.txt文件,有6列值
before_file = r'/mnt/d/Area_22/scene_1/scene_1_Origin.txt'
# 用代码修改后保存的.txt路径，有5列值
after_file = r'/mnt/d/Area_22/scene_1/scene_1_label.txt'

points = np.loadtxt(before_file)

for i in range(1, points.shape[1] - 4):
    # 找到最后一列不为NaN的值的索引
    valid_indices = ~np.isnan(points[:, -i])
    # 将最后一列不为NaN的值赋值给倒数第二列对应位置的值
    points[valid_indices, -(i + 1)] = points[valid_indices, -i]

np.savetxt(after_file, points[:, :5], fmt='%.8f')