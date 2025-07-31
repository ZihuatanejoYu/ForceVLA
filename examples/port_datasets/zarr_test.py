import zarr
import time
import numpy as np

# 1. 打开Zarr文件
z = zarr.open("/home/hairuo/sdp/data/zarr/test.zarr", mode="r")
front = z['data']['front_camera']
# z = zarr.open("/home/hairuo/robotwin/policy/DP/data/lift_pot-demo_randomized_x5-100.zarr", mode="r")
# front = z['data']['head_camera']

# 2. 打印数据集信息
print("=== Dataset Info ===")
print(f"Shape: {front.shape}")
print(f"Chunk size: {front.chunks}")
print(f"Compressor: {front.compressor}")
print(f"Data type: {front.dtype}")

# 3. 预热（首次访问会有额外开销）
_ = front[0]

# 4. 测试随机访问性能
random_indices = np.random.randint(0, len(front), size=100)
start = time.time()
for i in random_indices:
    _ = front[i]
random_time = (time.time() - start)/100

# 5. 测试连续访问性能（更接近训练场景）
start = time.time()
for i in range(100):
    _ = front[i]
sequential_time = (time.time() - start)/100

# 6. 计算压缩率
compression_ratio = front.nbytes / front.nbytes_stored

# 7. 打印结果
print("\n=== Performance ===")
print(f"Random access: {random_time:.4f}s per frame")
print(f"Sequential access: {sequential_time:.4f}s per frame")
print(f"\n=== Compression ===")
print(f"Original size: {front.nbytes/1e6:.2f} MB")
print(f"Compressed size: {front.nbytes_stored/1e6:.2f} MB")
print(f"Compression ratio: {compression_ratio:.2f}x")

# 8. 可选：验证数据完整性
sample_frame = front[0]
assert sample_frame.shape == front.shape[1:], "Frame shape mismatch"
assert sample_frame.dtype == front.dtype, "Data type mismatch"