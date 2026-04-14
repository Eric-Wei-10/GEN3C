import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from scipy.spatial import KDTree
from sklearn.decomposition import PCA


def get_top_percent_element(rgb_variance_map, visible_mask, percent=0.2):
    """
    参数:
    rgb_variance_map: H*W np.float32 array
    visible_mask: H*W bool array
    percent: 前百分之多少 (0.0 ~ 1.0)
    largest: True表示取最大的前20%，False表示取最小的前20%
    
    返回:
    selected_values: 选中的数值
    selected_indices: 选中的数值对应的原始坐标 tuple(rows, cols)
    """
    # 1. 获取 mask 区域内的数值
    # 注意：boolean indexing 返回的是一维数组
    valid_values = rgb_variance_map[visible_mask]

    # 2. 获取 mask 区域对应的原始坐标
    # np.where(condition) 返回 tuple(row_indices, col_indices)，顺序与 valid_values 严格对应（C-style row-major）
    valid_rows, valid_cols = np.where(visible_mask)

    # 如果 mask 区域为空，直接返回空
    if valid_values.size == 0:
        return np.array([]), (np.array([]), np.array([]))
    
    # 3. 计算需要选取的数量
    k = int(valid_values.size * percent)
    if k == 0:
        k = 1 # 至少取1个，或者根据需求处理 k=0 的情况
    
    # 4. 获取排序后的索引 (argsort 返回的是 valid_values 内部的索引)
    sorted_idx = np.argsort(valid_values)

    # 5. 取最大的前 20% (即排序后的最后 k 个，反转一下)
    top_k_idx = sorted_idx[::-1][:k]
        
    # 6. 利用局部索引映射回 数值 和 原始坐标
    selected_values = valid_values[top_k_idx]
    selected_rows = valid_rows[top_k_idx]
    selected_cols = valid_cols[top_k_idx]
    
    return selected_values, (selected_rows, selected_cols)

def convert_edge_mask_to_coords_list(edge_mask):
    coords_list = [tuple(x) for x in np.argwhere(edge_mask == 255)]
    return coords_list

def filter_high_depth_changes(depth_map, threshold=0.5, occlusion_mask=None):
    """
    过滤出深度图中变化剧烈的点
    :param depth_map: H*W 的 numpy 数组 (float32 或 uint16)
    :param threshold: 梯度幅值的阈值，取决于深度图的单位和量程
    :return: 掩码图 (mask)，变化剧烈的点为 255，其余为 0
    """

    # 1. 预处理：如果深度图包含噪声，可以先进行轻微的高斯模糊
    # 深度图建议使用中值滤波，因为它能更好地保留边缘同时去除噪声
    smoothed_depth = cv2.medianBlur(depth_map.astype(np.float32), 3)

    # 2. 计算 X 和 Y 方向的 Sobel 梯度
    # cv2.CV_64F 保证计算精度，避免溢出
    laplacian = cv2.Laplacian(smoothed_depth, cv2.CV_32F, ksize=3)
    laplacian_abs = np.absolute(laplacian)
    
    laplacian_abs[occlusion_mask] = 0.0

    # 4. 根据阈值生成掩码
    # 深度变化大的地方 magnitude 会很高
    _, mask = cv2.threshold(laplacian_abs, threshold, 255, cv2.THRESH_BINARY)
    
    return laplacian_abs, mask.astype(np.uint8)

def get_value(candidate, rgb_variance_map, occlusion_mask):
    cand_y, cand_x = candidate
    
    # Get coordinates of all valid pixels in the mask
    y_coords, x_coords = np.where(occlusion_mask)
    vars_at_mask = rgb_variance_map[occlusion_mask]
    
    # Vectorized distance calculation
    distances = np.sqrt((y_coords - cand_y)**2 + (x_coords - cand_x)**2)
    
    # warp distances to [1,10]
    distances /= np.max(distances)
    distances = distances * 9 + 1.0  # now distances in [1,10]
    
    # Vectorized sum
    value = np.sum((1.0 / (distances + 1e-5)) * vars_at_mask)
    return value

def main():
    # --- 模拟数据测试 ---
    # 创建一个简单的深度图：左边深 10，右边深 50

    parser = argparse.ArgumentParser(
        description=(
            "Postprocess variance to generate data labels:\n"
            "camera_npz: we only use the depth map for frame 0.\n"
            "data_root: where to find the variance map and camera.npz file\n"
            "T: specifiy the frame T\n"
            "multi: variance calculated in multi mode or not.\n"
        )
    )

    parser.add_argument("--camera_npz", type=str, required=True)
    
    parser.add_argument("--data_root", type=str, required=True)
    
    parser.add_argument("--T", type=str, required=True)

    parser.add_argument("--multi", action="store_true")

    args = parser.parse_args()

    output_dir = os.path.join(args.data_root, "outputs_multi" if args.multi else "outputs")
    os.makedirs(output_dir, exist_ok=True)
    cam = np.load(args.camera_npz)
    depth0_np = cam["depth0"].astype(np.float32)
    depth0_np = depth0_np[2:-2, :]  # drop the first and last 2 rows to match with the data

    if args.multi:
        backward_data = np.load(output_dir + f"/combined_priority_backward_t{args.T}.npz", allow_pickle=True)
        # import pdb; pdb.set_trace()
        mask_name = "combined_mask" # combined_mask = False, pixels invisible in frame 0.
        combined_mask = backward_data[mask_name].astype(bool)
        combined_mask = combined_mask[2:-2, :]
        
        combined_mask_vis = (combined_mask * 255).astype(np.uint8)
        combined_mask_save_path = os.path.join(output_dir, "combined_mask.png")
        cv2.imwrite(combined_mask_save_path, combined_mask_vis)
        # occlusion_mask = ~combined_mask # occlusion_mask = True, pixels invisible in frame 0.
        occlusion_mask = backward_data["occlusion_mask"][2:-2, :].astype(bool)
        occlusion_mask_vis = (occlusion_mask * 255).astype(np.uint8)
        occlusion_mask_save_path = os.path.join(output_dir, "occlusion_mask.png")
        cv2.imwrite(occlusion_mask_save_path, occlusion_mask_vis)

        visible_mask = combined_mask ^ occlusion_mask
        visible_mask_vis = (visible_mask * 255).astype(np.uint8)
        visible_mask_save_path = os.path.join(output_dir, "visible_mask.png")
        cv2.imwrite(visible_mask_save_path, visible_mask_vis)

        assert combined_mask.shape == (540, 720)
        assert occlusion_mask.shape == (540, 720)
        assert visible_mask.shape == (540, 720)

        # 执行过滤
        mag, edge_mask = filter_high_depth_changes(depth0_np, threshold=1.0, occlusion_mask=occlusion_mask)
        coords_list = convert_edge_mask_to_coords_list(edge_mask)

        variance_map = backward_data['variance_map'][:, 2:-2, :]
        rgb_variance_map = np.mean(variance_map, axis=0)
        assert variance_map.shape == (3, 540, 720)
        assert rgb_variance_map.shape == (540, 720)

        mean_variance_map_vis = (rgb_variance_map - np.min(rgb_variance_map)) / (np.max(rgb_variance_map) - np.min(rgb_variance_map) + 1e-8) * 255
        mean_variance_map_save_path = os.path.join(output_dir, "rgb_variance_map.png")
        cv2.imwrite(mean_variance_map_save_path, mean_variance_map_vis.astype(np.uint8))

        for coord in coords_list:
            occlusion_mask[coord] = False
        
        num_occluded_pixels = np.sum(occlusion_mask)
        
        value_map = np.zeros_like(rgb_variance_map)

        splat_radius = 10
        for candidate in coords_list:
            value = get_value(candidate, rgb_variance_map, occlusion_mask)
            value_map[candidate[0]-splat_radius:candidate[0]+splat_radius+1, candidate[1]-splat_radius:candidate[1]+splat_radius+1] += value
        
        # value_map /= num_occluded_pixels
        # value_map[visible_mask] = rgb_variance_map[visible_mask]
        
        # TODO: visible variance logic

        # visualize the value map
        value_map_vis = (value_map - np.min(value_map)) / (np.max(value_map) - np.min(value_map) + 1e-8) * 255
        value_map_save_path = os.path.join(output_dir, "value_map.png")
        cv2.imwrite(value_map_save_path, value_map_vis.astype(np.uint8))
        
        # the data root is the name of the scene
        file_name = os.path.basename(os.path.normpath(args.data_root))
        # save value map npz
        value_map_npz_path = os.path.join(f"{args.data_root}/../../weighted_mask", f"{file_name}.npz")
        
        # 把gt的weighted_mask load上来进行比对(临时代码，之后会删掉)
        folder_name = os.path.basename(os.path.dirname(output_dir))
        base_dir = "/cluster/project/cvg/students/shangwu/Pytorch-UNet/Actmap_gt_1000/weighted_mask"
        new_path = os.path.join(base_dir, f"{folder_name}.npz")
        gt_value_map = np.load(new_path)['weights']

        gt_value_map_vis = (gt_value_map - np.min(gt_value_map)) / (np.max(gt_value_map) - np.min(gt_value_map) + 1e-8) * 255
        gt_value_map_save_path = os.path.join(output_dir, "gt_value_map.png")
        cv2.imwrite(gt_value_map_save_path, gt_value_map_vis.astype(np.uint8))

        # import pdb; pdb.set_trace()
        os.makedirs(os.path.dirname(value_map_npz_path), exist_ok=True)
        np.savez(value_map_npz_path, weights=value_map)

        # --- 可视化 ---
        plt.figure(figsize=(12, 4))
        plt.subplot(131), plt.title("Original Depth"), plt.imshow(depth0_np, cmap='gray')
        plt.subplot(132), plt.title("Gradient Magnitude"), plt.imshow(mag, cmap='gray')
        plt.subplot(133), plt.title("High Change Points (Mask)"), plt.imshow(edge_mask, cmap='gray')
        
        plt_save_path = os.path.join(output_dir, "comparison_plot.png")
        plt.savefig(plt_save_path, dpi=300) # dpi=300 保证高清晰度
        plt.close() # 记得关闭，防止大量绘图占用内存
        
        print(f"所有结果已保存至: {os.path.abspath(output_dir)}")
        return
    else:
        raise NotImplementedError("single mode not implemented")
    # mask_name = "intersection_mask"
    # # mask_name = "combined_mask"
    # occlusion_mask = forward_data[mask_name].astype(np.float32) < 0.5
    # occlusion_mask = occlusion_mask[2:-2, :]
    
    # occlusion_mask_vis = (occlusion_mask * 255).astype(np.uint8)
    # occlusion_mask_save_path = os.path.join(output_dir, "occlusion_mask.png")
    # cv2.imwrite(occlusion_mask_save_path, occlusion_mask_vis)

    # visible_mask = forward_data[mask_name].astype(np.float32) >= 0.5
    # visible_mask = visible_mask[2:-2, :]
    
    visible_mask_vis = (visible_mask * 255).astype(np.uint8)
    visible_mask_save_path = os.path.join(output_dir, "visible_mask.png")
    cv2.imwrite(visible_mask_save_path, visible_mask_vis)
    
    # 执行过滤
    mag, edge_mask = filter_high_depth_changes(depth0_np, threshold=1.0, occlusion_mask=occlusion_mask)
    coords_list = convert_edge_mask_to_coords_list(edge_mask)
    
    # 2. Extract the specific arrays you need
    variance_map_all = backward_data['var_map'][:, 2:-2, :]  # This is the C*544*720 array, drop the first and last 2 rows to match for each frame

    ##### Eric start #####
    # import pdb; pdb.set_trace()

    C, _, _ = variance_map_all.shape
    if C == 768:
        # This is for dino variance
        
        data_transposed = variance_map_all.transpose(1, 2, 0)

        h, w, c = data_transposed.shape
        data_reshaped = data_transposed.reshape(-1, c)

        pca = PCA(n_components=3)
        data_pca = pca.fit_transform(data_reshaped)

        result_spatial = data_pca.reshape(h, w, 3)
        result_final = result_spatial.transpose(2, 0, 1)

        variance_map = np.mean(result_final, axis=0)
        assert variance_map.shape == (540, 720)

        # save variance map for visualization
        variance_map_vis = (variance_map - np.min(variance_map)) / (np.max(variance_map) - np.min(variance_map) + 1e-8) * 255
        variance_map_save_path = os.path.join(output_dir, "variance_map.png")
        cv2.imwrite(variance_map_save_path, variance_map_vis.astype(np.uint8))

        # import pdb; pdb.set_trace()

        return

    ##### Eric end #####
    
    for coord in coords_list:
        occlusion_mask[coord] = False
        visible_mask[coord] = False

    # 3. Take the red channel (first channel)
    # variance_map = variance_map_all[0]
    variance_map = np.mean(variance_map_all, axis=0)
    assert variance_map.shape == (540, 720)
    # save variance map for visualization
    variance_map_vis = (variance_map - np.min(variance_map)) / (np.max(variance_map) - np.min(variance_map) + 1e-8) * 255
    variance_map_save_path = os.path.join(output_dir, "variance_map.png")
    cv2.imwrite(variance_map_save_path, variance_map_vis.astype(np.uint8))
    
    # 4. Create a matrix to host the values
    value_map = np.zeros_like(variance_map)
    
    
    # the original weight mask contains labels in the shape of paint splats, we want to mimick that 
    splat_radius = 5
    for candidate in coords_list:
        value = get_value(candidate, variance_map, occlusion_mask)
        value_map[candidate[0]-splat_radius:candidate[0]+splat_radius+1, candidate[1]-splat_radius:candidate[1]+splat_radius+1] += value
        # value_map[candidate] = value
    


    
    # import pdb; pdb.set_trace()
    
    # 
    # selected_values, (selected_rows, selected_cols) = get_top_percent_element(variance_map, visible_mask, 0.2)
    # value_map[selected_rows, selected_cols] = selected_values

    # visualize the value map
    value_map_vis = (value_map - np.min(value_map)) / (np.max(value_map) - np.min(value_map) + 1e-8) * 255
    value_map_save_path = os.path.join(output_dir, "value_map.png")
    cv2.imwrite(value_map_save_path, value_map_vis.astype(np.uint8))
    
    # the data root is the name of the scene
    file_name = os.path.basename(os.path.normpath(args.data_root))
    # save value map npz
    value_map_npz_path = os.path.join(f"{args.data_root}/../../weighted_mask", f"{file_name}.npz")
    
    os.makedirs(os.path.dirname(value_map_npz_path), exist_ok=True)
    np.savez(value_map_npz_path, weights=value_map)

    # --- 可视化 ---
    plt.figure(figsize=(12, 4))
    plt.subplot(131), plt.title("Original Depth"), plt.imshow(depth0_np, cmap='gray')
    plt.subplot(132), plt.title("Gradient Magnitude"), plt.imshow(mag, cmap='gray')
    plt.subplot(133), plt.title("High Change Points (Mask)"), plt.imshow(edge_mask, cmap='gray')
    
    plt_save_path = os.path.join(output_dir, "comparison_plot.png")
    plt.savefig(plt_save_path, dpi=300) # dpi=300 保证高清晰度
    plt.close() # 记得关闭，防止大量绘图占用内存
    
    print(f"所有结果已保存至: {os.path.abspath(output_dir)}")


    

if __name__ == "__main__":
    main()
