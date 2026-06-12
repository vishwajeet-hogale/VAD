from .openlanev2_vad_dataset import VADOpenLaneV2Dataset


try:
    from .nuscenes_vad_dataset import VADCustomNuScenesDataset
except ModuleNotFoundError as exc:
    if exc.name != 'mmdet3d.ops.roiaware_pool3d':
        raise
    VADCustomNuScenesDataset = None


__all__ = [
    'VADOpenLaneV2Dataset'
]

if VADCustomNuScenesDataset is not None:
    __all__.append('VADCustomNuScenesDataset')
