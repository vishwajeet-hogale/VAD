import os

import mmcv
import numpy as np
from mmdet.datasets import DATASETS
from mmdet3d.datasets import Custom3DDataset
from pyquaternion import Quaternion


@DATASETS.register_module()
class VADOpenLaneV2Dataset(Custom3DDataset):
    """Inference-first OpenLane-V2 adapter for VAD.

    This adapter intentionally exposes only the camera and ego-pose metadata
    needed for BEV feature extraction. It does not attempt to convert OpenLane-V2
    annotations into VAD's training targets yet.
    """

    CAMS = (
        'ring_front_center',
        'ring_front_left',
        'ring_front_right',
        'ring_rear_left',
        'ring_rear_right',
        'ring_side_left',
        'ring_side_right',
    )

    def __init__(self, data_root, ann_file, **kwargs):
        super().__init__(data_root=data_root, ann_file=ann_file, **kwargs)

    def load_annotations(self, ann_file):
        data_infos = mmcv.load(ann_file, file_format='pkl')
        if isinstance(data_infos, dict):
            data_infos = list(data_infos.values())

        data_infos.sort(
            key=lambda info: (
                str(info.get('segment_id', '')),
                info.get('timestamp', 0),
            )
        )
        return data_infos

    def get_ann_info(self, index):
        return {}

    def get_data_info(self, index):
        info = self.data_infos[index]
        input_dict = dict(
            sample_idx=info['timestamp'],
            scene_token=info['segment_id'],
            lidar2global_rotation=np.array(info['pose']['rotation'], dtype=np.float32),
            lidar2global_translation=np.array(info['pose']['translation'], dtype=np.float32),
        )

        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []

            for cam_name in self.CAMS:
                cam_info = info['sensor'].get(cam_name)
                if cam_info is None:
                    continue

                image_paths.append(os.path.join(self.data_root, cam_info['image_path']))

                lidar2cam_r = np.linalg.inv(np.array(cam_info['extrinsic']['rotation']))
                lidar2cam_t = np.array(cam_info['extrinsic']['translation']) @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4, dtype=np.float32)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t

                intrinsic = np.array(cam_info['intrinsic']['K'], dtype=np.float32)
                viewpad = np.eye(4, dtype=np.float32)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = viewpad @ lidar2cam_rt.T

                lidar2img_rts.append(lidar2img_rt)
                cam_intrinsics.append(viewpad)
                lidar2cam_rts.append(lidar2cam_rt.T)

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    cam_intrinsic=cam_intrinsics,
                    lidar2cam=lidar2cam_rts,
                )
            )

        can_bus = np.zeros(18, dtype=np.float32)
        rotation = Quaternion._from_matrix(np.array(info['pose']['rotation']))
        can_bus[:3] = np.array(info['pose']['translation'], dtype=np.float32)
        can_bus[3:7] = np.array(rotation, dtype=np.float32)
        patch_angle = rotation.yaw_pitch_roll[0] / np.pi * 180
        if patch_angle < 0:
            patch_angle += 360
        can_bus[-2] = patch_angle / 180 * np.pi
        can_bus[-1] = patch_angle
        input_dict['can_bus'] = can_bus

        return input_dict