import os
import torch
import torch.nn as nn
import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# --- 解决 Matplotlib 中文显示乱码问题 ---
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# 1. 重新定义与训练时完全一致的模型结构
# ==========================================
class MLPModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLPModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, output_size - 1)
        )

    def forward(self, x):
        out_100 = self.network(x)
        first_point = out_100[:, 0:1]
        final_out = torch.cat([out_100, first_point], dim=1)
        return final_out


# ==========================================
# 2. 获取数据标准化器 (Scaler)
# ==========================================
def get_fitted_scaler(filepath='X.mat'):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"找不到 {filepath}！预测时需要它来对新输入进行正确的标准化。")

    mat_x = loadmat(filepath)
    x_key = [k for k in mat_x.keys() if not k.startswith('__')][0]
    X_data = mat_x[x_key]

    scaler = StandardScaler()
    scaler.fit(X_data)
    return scaler


# ==========================================
# 3. 核心预测函数
# ==========================================
def predict_gait_trajectory(model, scaler, new_features):
    features_np = np.array(new_features).reshape(1, -1)
    features_scaled = scaler.transform(features_np)
    features_tensor = torch.tensor(features_scaled, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        predicted_tensor = model(features_tensor)

    predicted_trajectory = predicted_tensor.squeeze().numpy()
    return predicted_trajectory


# ==========================================
# 4. 从 .mat 读取特征函数
# ==========================================
def load_new_subject_data(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"找不到 {filepath}！")

    mat_data = loadmat(filepath)
    data_key = [k for k in mat_data.keys() if not k.startswith('__')][0]
    raw_data = mat_data[data_key]

    processed_data = raw_data.flatten().reshape(1, -1)
    return processed_data


# ==========================================
# 5. 主程序：批量循环测试输入与绘图
# ==========================================
if __name__ == '__main__':
    weights_path = 'mlp_model_weights.pth'

    # 修改1：指定数据所在的统一文件夹
    data_dir = r'data\wu'
    # 确保输出文件夹存在
    os.makedirs(data_dir, exist_ok=True)

    # 修改2：定义需要批量处理的文件标号
    file_indices = [2, 3, 4, 5, 6]

    # 检查权重文件是否存在
    if not os.path.exists(weights_path):
        print(f"错误：找不到权重文件 '{weights_path}'！请确保你已经运行并保存了训练代码。")
        exit()

    # 初始化模型并加载权重
    print("正在加载模型权重...")
    model = MLPModel(input_size=9, output_size=101)
    model.load_state_dict(torch.load(weights_path))

    # 获取数据标准化器 (注意：X.mat 依然需要在这个 python 脚本的同级目录下)
    print("正在恢复数据标准化器...")
    scaler = get_fitted_scaler('X.mat')

    print("\n=== 开始批量预测步态轨迹 ===")

    for idx in file_indices:
        print(f"\n----------------------------------------")
        print(f"正在处理第 {idx} 组数据...")

        # 动态拼接输入和输出文件的路径
        new_data_path = os.path.join(data_dir, f'X_new_{idx}.mat')
        output_filename = os.path.join(data_dir, f'predict{idx}.csv')

        # 检查文件是否存在，防止某一组漏生成导致报错中断
        if not os.path.exists(new_data_path):
            print(f"⚠️ 找不到文件 {new_data_path}，已跳过。")
            continue

        # 1. 读取数据
        my_new_patient_features = load_new_subject_data(new_data_path)

        if my_new_patient_features.shape[1] != 9:
            print(f"⚠️ 第 {idx} 组特征数量错误！跳过该组。")
            continue

        speed_val = my_new_patient_features[0, 8]

        # 2. 生成网络预测轨迹
        raw_trajectory = predict_gait_trajectory(model, scaler, my_new_patient_features)

        # 3. S-G 平滑滤波去锯齿
        traj_100 = raw_trajectory[:-1]
        smoothed_100 = savgol_filter(traj_100, window_length=15, polyorder=3, mode='wrap')
        smoothed_trajectory = np.append(smoothed_100, smoothed_100[0])

        # 4. 保存为对应的 CSV 文件
        np.savetxt(output_filename, smoothed_trajectory, delimiter=',', fmt='%.6f', header='Smoothed_Trajectory',
                   comments='')
        print(f"✅ 预测轨迹已成功保存至: {output_filename}")
        print(f"首尾闭合验证：起点 {smoothed_trajectory[0]:.4f}，终点 {smoothed_trajectory[-1]:.4f}")

        # 5. 可视化这组结果
        plt.figure(figsize=(10, 5))
        plt.plot(raw_trajectory, color='gray', linestyle='--', linewidth=1.5, alpha=0.5, label='原始网络输出 (带锯齿)')
        plt.plot(smoothed_trajectory, color='magenta', linewidth=2.5, label='S-G平滑后轨迹 (无锯齿不失真)')
        plt.title(f'第 {idx} 组步态轨迹预测 (基于 X_new_{idx}.mat, 速度: {speed_val:.4f}m/s)')
        plt.xlabel('步态周期百分比 (1-101)')
        plt.ylabel('关节角度/坐标值')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.show()  # 每组出图后，需要关掉图表窗口才会继续跑下一组

    print("\n=== 所有批量预测任务已完成！ ===")