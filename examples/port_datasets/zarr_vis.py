import zarr
import argparse

def print_zarr_structure(group, indent=0):
    """递归打印 Zarr 组/数组结构"""
    prefix = "    " * indent
    for key, item in group.items():
        if isinstance(item, zarr.Group):
            print(f"{prefix}📁 {key}/ (Group)")
            print_zarr_structure(item, indent + 1)
        else:
            print(f"{prefix}🔢 {key} {item.shape} {item.dtype}")
            # 可选：打印属性
            if item.attrs:
                print(f"{prefix}   ├─ Attributes: {dict(item.attrs)}")

parser = argparse.ArgumentParser(description='Print the structure of a Zarr file')
parser.add_argument('--zarr_path', type=str, required=True,
                    help='Path to the Zarr file or directory')
args = parser.parse_args()

# 使用示例
zarr_data = zarr.open(args.zarr_path, mode="r")
print_zarr_structure(zarr_data)
