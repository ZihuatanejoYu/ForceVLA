import pdb
import dataclasses
import logging
import re
from typing import Protocol, runtime_checkable

import flax.traverse_util
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "s3://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        # Add all missing LoRA weights.
        return _merge_params(loaded_params, params, missing_regex=".*lora.*")


@dataclasses.dataclass(frozen=True)
class Pi0GuidanceWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "s3://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        # Add all missing LoRA weights.
        return _merge_params(loaded_params, params, missing_regex=".*lora.*|.*limoe.*|.*force.*")


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


# def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
#     """Merges the loaded parameters with the reference parameters.

#     Args:
#         loaded_params: The parameters to merge.
#         params: The reference parameters.
#         missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

#     Returns:
#         A new dictionary with the merged parameters.
#     """
#     flat_ref = flax.traverse_util.flatten_dict(params, sep="/") # initialized params
#     flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/") # checkpoint params

#     # First, take all weights that are a subset of the reference weights.
#     result = {}
#     for k, v in flat_loaded.items():
#         if k in flat_ref:
#             result[k] = v.astype(flat_ref[k].dtype) # registrate all checkpoint params

#     # Then, merge any missing weights as defined by the missing regex.
#     pdb.set_trace()
#     pattern = re.compile(missing_regex)
#     for k in {k for k in flat_ref if pattern.fullmatch(k)}: # select all missing_regex patterns in initialized params
#         if k not in result: # if missing_regex params not registrated
#             result[k] = flat_ref[k] # registrate it

#     return flax.traverse_util.unflatten_dict(result, sep="/")


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Optimized function to merge the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    # 编译正则表达式（只需编译一次）
    pattern = re.compile(missing_regex)
    
    # 扁平化字典
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")
    # pdb.set_trace()
    
    # 创建参考键的集合以加快查找，只需计算一次
    ref_keys = set(flat_ref.keys())
    loaded_keys = set(flat_loaded.keys())
    common_keys = loaded_keys.intersection(ref_keys)
    
    # 预先筛选需要添加的missing_regex模式的键
    missing_pattern_keys = {k for k in flat_ref if pattern.fullmatch(k) and k not in common_keys}
    
    # 一次性构建结果字典，避免多次迭代
    result = {}
    
    # 1. 添加所有共同键的参数（checkpoint参数）
    for k in common_keys:
        v = flat_loaded[k]
        ref_dtype = flat_ref[k].dtype
        # 只在必要时进行类型转换
        if v.dtype != ref_dtype:
            result[k] = v.astype(ref_dtype)
        else:
            result[k] = v
    
    # 2. 添加所有missing_regex模式的键（初始化参数）
    for k in missing_pattern_keys:
        result[k] = flat_ref[k]
    
    # 展开结果字典
    return flax.traverse_util.unflatten_dict(result, sep="/")

    