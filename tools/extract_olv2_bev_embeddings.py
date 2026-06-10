import argparse
import copy
import importlib
import os
import sys
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from mmcv import Config
from mmcv.parallel import DataContainer, scatter
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmcv.utils import import_modules_from_strings
from mmdet.datasets import replace_ImageToTensor
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from projects.mmdet3d_plugin.datasets.builder import build_dataloader  # noqa: E402


ALL_MODES = ['baseline_meanpool', 'pos_meanpool', 'channel_collapse_mean', 'coarse_grid']
DEFAULT_OUTPUT_DIR = '/scratch/hogale.v/GMM_embeddings'
POS_DIM = 64
COARSE_H = 20
COARSE_W = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Extract OpenLane-V2 BEV embeddings from VAD.')
    parser.add_argument(
        '--config',
        default='projects/configs/VAD/VAD_tiny_stage_1_olv2_extract.py',
        help='Path to the VAD OpenLane-V2 extraction config.',
    )
    parser.add_argument('--checkpoint', required=True, help='Path to the VAD checkpoint.')
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='train')
    parser.add_argument(
        '--modes',
        default='all',
        help="Comma-separated embedding modes or 'all'.",
    )
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--device', default=None)
    parser.add_argument('--bev-h', type=int, default=None)
    parser.add_argument('--bev-w', type=int, default=None)
    parser.add_argument('--disable-temporal', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def import_plugin_modules(cfg: Config, config_path: str) -> None:
    if cfg.get('custom_imports', None):
        import_modules_from_strings(**cfg.custom_imports)

    if hasattr(cfg, 'plugin') and cfg.plugin:
        if hasattr(cfg, 'plugin_dir'):
            module_dir = os.path.dirname(cfg.plugin_dir)
        else:
            module_dir = os.path.dirname(config_path)

        module_parts = [part for part in module_dir.split('/') if part]
        if not module_parts:
            return
        module_path = module_parts[0]
        for part in module_parts[1:]:
            module_path = f'{module_path}.{part}'
        importlib.import_module(module_path)


def resolve_device(cfg: Config, override: str = None) -> torch.device:
    if override is not None:
        return torch.device(override)
    if torch.cuda.is_available():
        gpu_ids = cfg.get('gpu_ids', [0])
        gpu_idx = int(gpu_ids[0]) if len(gpu_ids) > 0 else 0
        return torch.device(f'cuda:{gpu_idx}')
    return torch.device('cpu')


def select_split_cfg(cfg: Config, split: str):
    if split not in cfg.data:
        raise KeyError(f"Split '{split}' not found in cfg.data")

    split_cfg = copy.deepcopy(cfg.data[split])
    test_cfg = cfg.data.get('test', None)
    if isinstance(test_cfg, dict):
        if 'pipeline' in test_cfg:
            split_cfg.pipeline = copy.deepcopy(test_cfg.pipeline)
        elif 'dataset' in test_cfg and 'pipeline' in test_cfg.dataset:
            split_cfg.pipeline = copy.deepcopy(test_cfg.dataset.pipeline)
    elif isinstance(test_cfg, list) and len(test_cfg) > 0:
        first_test_cfg = test_cfg[0]
        if isinstance(first_test_cfg, dict) and 'pipeline' in first_test_cfg:
            split_cfg.pipeline = copy.deepcopy(first_test_cfg.pipeline)

    split_cfg.test_mode = True
    return split_cfg


def build_dataset_and_loader(cfg: Config, split: str):
    dataset_cfg = select_split_cfg(cfg, split)
    samples_per_gpu = dataset_cfg.pop('samples_per_gpu', 1)
    if samples_per_gpu > 1:
        dataset_cfg.pipeline = replace_ImageToTensor(dataset_cfg.pipeline)
    dataset = build_dataset(dataset_cfg)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.get('workers_per_gpu', 2),
        dist=False,
        shuffle=False,
    )
    return dataset, data_loader


def move_batch_to_device(batch_data: dict, device: torch.device) -> dict:
    if device.type == 'cuda':
        return scatter(batch_data, [device.index])[0]

    unwrapped = {}
    for key, value in batch_data.items():
        if isinstance(value, DataContainer):
            unwrapped[key] = value.data[0]
        else:
            unwrapped[key] = value
    return unwrapped


def _flatten_img_metas_structure(obj: Any) -> List[dict]:
    if isinstance(obj, DataContainer):
        obj = obj.data
    if isinstance(obj, tuple):
        obj = list(obj)
    while isinstance(obj, list) and len(obj) == 1 and isinstance(obj[0], (list, tuple)):
        obj = obj[0]
    if isinstance(obj, list) and len(obj) > 0 and isinstance(obj[0], dict):
        return obj

    out: List[dict] = []
    if isinstance(obj, dict):
        out.append(obj)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            out.extend(_flatten_img_metas_structure(item))
    return out


def extract_sample_ids(batch_data: dict, start_index: int) -> List[str]:
    if 'img_metas' not in batch_data:
        return []
    metas = _flatten_img_metas_structure(batch_data['img_metas'])
    sample_ids = []
    for i, meta in enumerate(metas):
        scene_token = str(meta.get('scene_token', 'unknown_scene'))
        sample_idx = meta.get('sample_idx', meta.get('timestamp', start_index + i))
        sample_ids.append(f'{scene_token}:{sample_idx}')
    return sample_ids


def make_2d_sincos_positional_encoding(h: int, w: int, dim: int) -> torch.Tensor:
    if dim % 4 != 0:
        raise ValueError(f'pos enc dim must be divisible by 4, got {dim}')

    half = dim // 2

    def _1d_pe(length: int, width: int) -> torch.Tensor:
        position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, width, 2, dtype=torch.float32) * (-np.log(10000.0) / width)
        )
        pe = torch.zeros(length, width, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    h_pe = _1d_pe(h, half)
    w_pe = _1d_pe(w, half)
    h_grid = h_pe.unsqueeze(1).expand(h, w, half)
    w_grid = w_pe.unsqueeze(0).expand(h, w, half)
    return torch.cat([h_grid, w_grid], dim=-1)


def bev_to_bhwc(tensor: torch.Tensor, bev_h: int, bev_w: int) -> torch.Tensor:
    hw = bev_h * bev_w
    if tensor.ndim == 3:
        if tensor.shape[1] == hw:
            batch, _, channels = tensor.shape
            return tensor.view(batch, bev_h, bev_w, channels).contiguous()
        if tensor.shape[0] == hw:
            _, batch, channels = tensor.shape
            return tensor.view(bev_h, bev_w, batch, channels).permute(2, 0, 1, 3).contiguous()
    if tensor.ndim == 4 and tensor.shape[1] == bev_h and tensor.shape[2] == bev_w:
        return tensor.contiguous()
    raise ValueError(f'Unsupported BEV tensor shape {tuple(tensor.shape)} for ({bev_h}, {bev_w})')


def derive_mode_features(
    bhwc: torch.Tensor,
    pos_enc_hwd: torch.Tensor,
    selected_modes: List[str],
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    batch, bev_h, bev_w, channels = bhwc.shape

    if 'baseline_meanpool' in selected_modes:
        out['baseline_meanpool'] = bhwc.mean(dim=(1, 2)).numpy().astype(np.float32)

    if 'pos_meanpool' in selected_modes:
        pos = pos_enc_hwd.unsqueeze(0).expand(batch, bev_h, bev_w, pos_enc_hwd.shape[-1])
        feat = torch.cat([bhwc, pos], dim=-1).mean(dim=(1, 2))
        out['pos_meanpool'] = feat.numpy().astype(np.float32)

    if 'channel_collapse_mean' in selected_modes:
        feat = bhwc.mean(dim=-1).reshape(batch, bev_h * bev_w)
        out['channel_collapse_mean'] = feat.numpy().astype(np.float32)

    if 'coarse_grid' in selected_modes:
        bchw = bhwc.permute(0, 3, 1, 2).contiguous()
        coarse = F.adaptive_avg_pool2d(bchw, output_size=(COARSE_H, COARSE_W))
        coarse = coarse.permute(0, 2, 3, 1).contiguous()
        out['coarse_grid'] = coarse.numpy().astype(np.float32)

    return out


def update_temporal_state(img_metas: List[dict], state: Dict[str, Any]) -> List[dict]:
    current_metas = copy.deepcopy(img_metas)
    if len(current_metas) != 1:
        raise ValueError('Temporal extraction currently expects batch_size=1.')

    current_meta = current_metas[0]
    scene_token = current_meta['scene_token']
    if scene_token != state['scene_token']:
        state['prev_bev'] = None
        state['scene_token'] = scene_token

    tmp_pos = copy.deepcopy(current_meta['can_bus'][:3])
    tmp_angle = copy.deepcopy(current_meta['can_bus'][-1])
    if state['prev_bev'] is not None:
        current_meta['can_bus'][:3] -= state['prev_pos']
        current_meta['can_bus'][-1] -= state['prev_angle']
    else:
        current_meta['can_bus'][:3] = 0
        current_meta['can_bus'][-1] = 0

    state['next_pos'] = tmp_pos
    state['next_angle'] = tmp_angle
    return current_metas


def finalize_temporal_state(state: Dict[str, Any], prev_bev: torch.Tensor) -> None:
    state['prev_bev'] = prev_bev.detach()
    state['prev_pos'] = state.pop('next_pos')
    state['prev_angle'] = state.pop('next_angle')


def main() -> None:
    args = parse_args()

    if args.modes.lower() == 'all':
        selected_modes = list(ALL_MODES)
    else:
        selected_modes = [mode.strip() for mode in args.modes.split(',') if mode.strip()]
        for mode in selected_modes:
            if mode not in ALL_MODES:
                raise ValueError(f'Unknown mode: {mode}. Valid: {ALL_MODES}')

    os.makedirs(args.output_dir, exist_ok=True)

    cfg = Config.fromfile(args.config)
    import_plugin_modules(cfg, args.config)
    cfg.model.pretrained = None
    device = resolve_device(cfg, args.device)
    bev_h = args.bev_h or int(cfg.model.pts_bbox_head.bev_h)
    bev_w = args.bev_w or int(cfg.model.pts_bbox_head.bev_w)

    _, data_loader = build_dataset_and_loader(cfg, args.split)

    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)

    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']

    model.to(device)
    model.eval()

    pos_enc = make_2d_sincos_positional_encoding(bev_h, bev_w, POS_DIM)
    accum: Dict[str, List[np.ndarray]] = {mode: [] for mode in selected_modes}
    all_sample_ids: List[str] = []

    temporal_state = dict(
        prev_bev=None,
        scene_token=None,
        prev_pos=0,
        prev_angle=0,
    )

    processed = 0
    max_items = args.max_samples or len(data_loader)

    with torch.no_grad():
        iterator = tqdm(data_loader, total=len(data_loader), desc=f'Extracting {args.split}')
        for data in iterator:
            if args.max_samples is not None and processed >= args.max_samples:
                break

            batch_ids = extract_sample_ids(data, start_index=processed)
            inputs = move_batch_to_device(data, device)

            img = inputs['img']
            img_metas = inputs['img_metas']
            prev_bev = None
            if not args.disable_temporal:
                img_metas = update_temporal_state(img_metas, temporal_state)
                prev_bev = temporal_state['prev_bev']

            img_feats = model.extract_feat(img=img, img_metas=img_metas)
            bev = model.pts_bbox_head(
                img_feats,
                img_metas,
                prev_bev=prev_bev,
                only_bev=True,
            )
            bhwc = bev_to_bhwc(bev.detach().float().cpu(), bev_h, bev_w)

            if args.dry_run:
                features = derive_mode_features(bhwc, pos_enc, selected_modes)
                for mode, array in features.items():
                    print(f'[DEBUG] mode={mode}, batch_shape={array.shape}')
                print(f'[INFO] Dry run OK. split={args.split} device={device}')
                return

            batch = bhwc.shape[0]
            if len(batch_ids) == 0:
                batch_ids = [f'sample_{processed + i}' for i in range(batch)]
            if len(batch_ids) != batch:
                keep = min(len(batch_ids), batch)
                batch_ids = batch_ids[:keep]
                bhwc = bhwc[:keep]
                batch = keep

            if args.max_samples is not None:
                remaining = args.max_samples - processed
                if batch > remaining:
                    bhwc = bhwc[:remaining]
                    batch_ids = batch_ids[:remaining]
                    batch = remaining

            features = derive_mode_features(bhwc, pos_enc, selected_modes)
            for mode, array in features.items():
                accum[mode].append(array)
            all_sample_ids.extend(batch_ids)
            processed += batch
            iterator.set_postfix(processed=processed, limit=max_items)

            if not args.disable_temporal:
                finalize_temporal_state(temporal_state, bev)

    if processed == 0:
        raise RuntimeError('No samples processed.')

    sample_ids_arr = np.array(all_sample_ids, dtype=object)
    for mode in selected_modes:
        embeddings = np.concatenate(accum[mode], axis=0).astype(np.float32)
        output_path = os.path.join(args.output_dir, f'bev_embeddings_{args.split}_{mode}.npz')
        np.savez(
            output_path,
            embeddings=embeddings,
            sample_ids=sample_ids_arr,
            mode=np.array(mode, dtype=object),
        )
        print(f'[INFO] {mode}: saved {embeddings.shape} -> {output_path}')


if __name__ == '__main__':
    main()