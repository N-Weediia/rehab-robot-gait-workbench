import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- 解决 Matplotlib 中文显示乱码问题 ---
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def align_and_save(input_path, output_path, idx):
    """
    读取轨迹，以最小值为起点重新拼接，并保持 101 个点首尾闭合
    """
    if not os.path.exists(input_path):
        print(f"⚠️ 找不到文件 {input_path}，已跳过。")
        return False

    # 读取原始数据
    df = pd.read_csv(input_path)
    # 获取列名 (通常是 'Smoothed_Trajectory')
    col_name = df.columns[0]
    traj = df[col_name].values

    # 1. 找到最小值及其索引
    min_idx = np.argmin(traj)

    # 2. 剥离最后一个重复点，提取核心的 100 个点 (索引 0~99)
    # 因为原始轨迹是首尾相连的 101 个点 (traj[0] == traj[-1])
    core_traj = traj[:-1]

    # 如果最小值刚好出现在最后一个点(索引100)，等同于索引0
    if min_idx == len(traj) - 1:
        min_idx = 0

    # 3. 进行切片和拼接 (去重拼接)
    # 后半段：从最小值开始，一直到核心周期的末尾
    part2 = core_traj[min_idx:]
    # 前半段：从核心周期的开头，一直到最小值的前一个点
    part1 = core_traj[:min_idx]

    # 拼接成新的核心周期 (恰好还是 100 个独立点)
    new_core = np.concatenate((part2, part1))

    # 4. 补齐最后一位 (使首尾再次闭环，长度恢复到 101)
    # 新核心的第 0 个点就是最小值，我们把它复制一份追加到末尾
    new_traj = np.append(new_core, new_core[0])

    # 5. 【新增】极值归一化：剥离原有的物理量纲，只保留 0~1 的纯形状！
    t_min = np.min(new_traj)
    t_max = np.max(new_traj)
    shape_traj = (new_traj - t_min) / (t_max - t_min)

    # 保存为新的 CSV 文件
    np.savetxt(output_path, shape_traj, delimiter=',', fmt='%.6f', header=col_name, comments='')
    print(f"✅ 第 {idx} 组处理完成！最小值原索引为 {min_idx}，已平移至 0 并归一化为 0~1。已保存至: {output_path}")

    return traj, shape_traj


if __name__ == '__main__':
    # 配置文件夹路径 (请根据你的实际路径修改)
    # data_dir = r'E:\PycharmProjects\data\zou01'
    data_dir = r'data\wu'
    file_indices = [2, 3, 4, 5, 6]

    print("=== 开始批量对齐轨迹相位 ===")

    plt.figure(figsize=(15, 10))

    for i, idx in enumerate(file_indices):
        input_file = os.path.join(data_dir, f'predict{idx}.csv')
        # 为了不覆盖原始数据，我们存为 _aligned.csv
        output_file = os.path.join(data_dir, f'predict{idx}_aligned.csv')

        result = align_and_save(input_file, output_file, idx)

        if result:
            original_traj, aligned_traj = result

            # 画图对比，直观验证平移效果
            plt.subplot(3, 2, i + 1)
            plt.plot(original_traj, color='gray', linestyle='--', label='原始轨迹 (起点非最低点)')
            plt.plot(aligned_traj, color='blue', linewidth=2, label='对齐后轨迹 (起点=终点=最小值)')

            # 标出最小值点作为视觉确认
            plt.plot(0, aligned_traj[0], 'ro', markersize=6, label='新的起点 (索引0)')
            plt.plot(100, aligned_traj[-1], 'ro', markersize=6)

            plt.title(f'第 {idx} 组 (predict{idx}.csv) 相位平移验证')
            plt.xlabel('步态周期百分比 (0-100)')
            plt.ylabel('关节角度')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()
    print("\n=== 所有文件相位对齐完毕！ ===")