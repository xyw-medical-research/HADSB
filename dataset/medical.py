"""
Medical dataset loader for T1/T2/PET modality conversion
Handles paired T1 (LAVA), T2, and PET data
"""

import glob
import json
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import torch
from torch.utils.data import Dataset

# Import region/organ configuration
try:
    from configs.region_organ_config import (
        BODY_PARTS, ORGANS, 
        BODY_PART_TO_IDX, ORGAN_TO_IDX,
        NUM_BODY_PARTS, NUM_ORGANS,
        get_body_part_idx, create_organ_mask,
    )
    _REGION_CONFIG_AVAILABLE = True
except ImportError:
    _REGION_CONFIG_AVAILABLE = False
    NUM_BODY_PARTS = 11
    NUM_ORGANS = 36



class MedicalModalityDataset(Dataset):
    """Dataset for medical modality conversion (T1->T2 with PET conditioning)"""
    
    def __init__(
        self,
        data_dir,
        split='train',
        image_size=512,
        normalize=True,
        pet_log_scale: bool = False,
        pet_log_eps: float = 1e-6,
        pet_norm: str = "max",
        # LAVA water/fat conditioning
        use_lava_water: bool = False,
        use_lava_fat: bool = False,
        # Flat directory structure (no train/val subfolders)
        flat_structure: bool = False,
        # Validation groups for flat structure
        val_patients: Optional[List[str]] = None,
        # Body Part / Organ semantic conditioning (JSON-based)
        use_body_part_condition: bool = False,
        use_organ_crossattn: bool = False,
        region_info_file: Optional[str] = None,
    ):
        """
        Args:
            data_dir: Root directory containing train/val folders
            split: 'train' or 'val'
            image_size: Target size for images (default 512)
            normalize: Whether to normalize to [-1, 1]
            pet_log_scale: Whether to apply log scaling to PET inputs
            pet_log_eps: Numerical epsilon to guard PET log scaling
            pet_norm: PET normalization method ("max", "p1p99", "zscore")
            use_lava_water: Whether to load LAVA water images
            use_lava_fat: Whether to load LAVA fat images
            flat_structure: If True, data_dir directly contains modality folders (no train/val)
            val_patients: Validation group identifiers used when flat_structure=True
            use_body_part_condition: Whether to use Body Part semantic embedding (from JSON)
            use_organ_crossattn: Whether to use Organ cross-attention (from JSON)
            region_info_file: Path to JSON file with body part and organ annotations
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.val_patients = set(val_patients) if val_patients else set()
        self.image_size = image_size
        self.normalize = normalize
        self.pet_log_scale = pet_log_scale
        self.pet_log_eps = pet_log_eps
        self.pet_norm = pet_norm
        
        # LAVA water/fat conditioning
        self.use_lava_water = use_lava_water
        self.use_lava_fat = use_lava_fat
        self.flat_structure = flat_structure
        
        # Body Part / Organ semantic conditioning (NEW)
        self.use_body_part_condition = use_body_part_condition
        self.use_organ_crossattn = use_organ_crossattn
        self.region_info: Dict[str, Any] = {}
        
        if (use_body_part_condition or use_organ_crossattn) and region_info_file:
            self._load_region_info(region_info_file)
        
        # Build file lists for each modality
        if flat_structure:
            # Flat structure: data_dir/LAVA, data_dir/T2, etc.
            self.t1_dir = self.data_dir / 'lava'
            self.t2_dir = self.data_dir / 'T2' 
            self.pet_dir = self.data_dir / 'PET'
            self.lava_water_dir = self.data_dir / 'lava_water' if use_lava_water else None
            self.lava_fat_dir = self.data_dir / 'lava_fat' if use_lava_fat else None
        else:
            # Original structure: data_dir/split/LAVA, etc.
            self.t1_dir = self.data_dir / split / 'LAVA'
            self.t2_dir = self.data_dir / split / 'T2' 
            self.pet_dir = self.data_dir / split / 'PET'
            self.lava_water_dir = self.data_dir / split / 'lava_water' if use_lava_water else None
            self.lava_fat_dir = self.data_dir / split / 'lava_fat' if use_lava_fat else None
        
        # Get all T1 files and extract base names
        all_t1_files = sorted(glob.glob(str(self.t1_dir / '*.npy')))
        
        # Filter based on split when using flat structure with val groups.
        if flat_structure and self.val_patients:
            filtered_files = []
            for f in all_t1_files:
                # Extract group id from filename prefix: "<group>_<case>_sliceXXX.npy".
                base = Path(f).stem
                group_id = '_'.join(base.split('_')[:2])

                is_val_patient = group_id in self.val_patients
                if split == 'val' and is_val_patient:
                    filtered_files.append(f)
                elif split == 'train' and not is_val_patient:
                    filtered_files.append(f)
            self.t1_files = filtered_files
        else:
            self.t1_files = all_t1_files
        
        self.base_names = [Path(f).stem for f in self.t1_files]
        
        # Verify paired data exists
        self.verify_paired_data()
        
        print(f"[MedicalDataset] Loaded {len(self.t1_files)} {split} samples")
        print(f"[MedicalDataset] Image size: {image_size}x{image_size}")
        if self.pet_log_scale:
            print(f"[MedicalDataset] PET log scaling enabled (eps={self.pet_log_eps:.1e})")
        if self.pet_norm and self.pet_norm != "max":
            print(f"[MedicalDataset] PET normalization: {self.pet_norm}")
        if use_lava_water:
            print(f"[MedicalDataset] LAVA water conditioning enabled")
        if use_lava_fat:
            print(f"[MedicalDataset] LAVA fat conditioning enabled")
        if use_body_part_condition:
            print(f"[MedicalDataset] Body Part semantic conditioning enabled ({len(self.region_info)} annotations)")
        if use_organ_crossattn:
            print(f"[MedicalDataset] Organ cross-attention enabled")
    
    def _load_region_info(self, region_info_file: str):
        """
        Load body part and organ annotations from JSON file.
        
        Expected JSON format:
        {
            "sample_slice016.png": {
                "Body Part": "Hips",
                "Organs Present": ["Bladder", "Rectum", "Pelvic Bone", ...]
            },
            ...
        }
        """
        region_path = Path(region_info_file)
        if not region_path.exists():
            print(f"[MedicalDataset] Warning: region_info_file not found: {region_info_file}")
            return
        
        with open(region_path, 'r') as f:
            raw_data = json.load(f)
        
        # Normalize keys: remove .png extension, handle list-wrapped values
        for key, value in raw_data.items():
            # Remove .png extension if present
            normalized_key = key.replace('.png', '')
            
            # Handle list-wrapped values (some entries might be [{"Body Part": ...}])
            if isinstance(value, list) and len(value) > 0:
                value = value[0]
            
            if isinstance(value, dict):
                self.region_info[normalized_key] = value
        
        print(f"[MedicalDataset] Loaded region info for {len(self.region_info)} samples")
    
    def _get_body_part_id(self, base_name: str) -> int:
        """Get body part index for a sample."""
        if not _REGION_CONFIG_AVAILABLE:
            return 0
        
        info = self.region_info.get(base_name, {})
        body_part = info.get('Body Part', '')
        return get_body_part_idx(body_part) if body_part else 0
    
    def _get_organ_mask(self, base_name: str) -> List[int]:
        """Get multi-hot organ mask for a sample."""
        if not _REGION_CONFIG_AVAILABLE:
            return [0] * NUM_ORGANS
        
        info = self.region_info.get(base_name, {})
        organs = info.get('Organs Present', [])
        return create_organ_mask(organs)
    
    def verify_paired_data(self):
        """Check that all modalities have matching files"""
        missing_files = []
        for base in self.base_names[:10]:  # Check first 10 as sample
            t2_path = self.t2_dir / f"{base}.npy"
            pet_path = self.pet_dir / f"{base}.npy"
            if not t2_path.exists():
                missing_files.append(f"T2: {t2_path}")
            if not pet_path.exists():
                missing_files.append(f"PET: {pet_path}")
            if self.use_lava_water and self.lava_water_dir:
                water_path = self.lava_water_dir / f"{base}.npy"
                if not water_path.exists():
                    missing_files.append(f"lava_water: {water_path}")
            if self.use_lava_fat and self.lava_fat_dir:
                fat_path = self.lava_fat_dir / f"{base}.npy"
                if not fat_path.exists():
                    missing_files.append(f"lava_fat: {fat_path}")
        
        if missing_files:
            print(f"Warning: Missing files detected:")
            for f in missing_files[:5]:  # Show first 5
                print(f"  {f}")
            print(f"... and {len(missing_files)-5} more" if len(missing_files) > 5 else "")
            # Don't raise error, just warn
    
    def __len__(self):
        return len(self.t1_files)
    
    def load_and_preprocess(self, path, modality='mri'):
        """Load and preprocess a single image"""
        if not path.exists():
            # Create a zero image as fallback
            print(f"Warning: File not found {path}, using zeros")
            data = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        else:
            data = np.load(path)
        
        # Handle different data types
        data = data.astype(np.float32)

        # Check and sanitize NaN/Inf values.
        if np.isnan(data).any() or np.isinf(data).any():
            print(f"Warning: NaN/Inf found in {path}, replacing with 0")
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        # Allow optional log scaling for PET modality before normalization
        if modality == 'pet' and self.pet_log_scale:
            data = np.clip(data, a_min=0.0, a_max=None)
            data = np.log1p(data)
        
        
        # Resize from 1024x1024 to target size (512x512)
        if data.shape != (self.image_size, self.image_size):
            # Simple downsampling by taking every other pixel
            if data.shape[0] == 1024 and self.image_size == 512:
                data = data[::2, ::2]
            else:
                # More general resize using torch
                data_tensor = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)
                data_tensor = torch.nn.functional.interpolate(
                    data_tensor, 
                    size=(self.image_size, self.image_size),
                    mode='bilinear',
                    align_corners=False
                )
                data = data_tensor.squeeze().numpy()
        
        # Normalize to [0, 1] first
        # Use a minimum threshold to avoid instability from near-zero divisors.
        if modality == 'pet' and self.pet_norm in ("p1p99", "zscore"):
            if self.pet_norm == "p1p99":
                p1 = np.percentile(data, 1)
                p99 = np.percentile(data, 99)
                denom = p99 - p1
                if denom > self.pet_log_eps:
                    data = (data - p1) / denom
                    data = np.clip(data, 0.0, 1.0)
                else:
                    data = np.zeros_like(data)
            elif self.pet_norm == "zscore":
                mean = float(data.mean())
                std = float(data.std())
                if std > self.pet_log_eps:
                    z = (data - mean) / std
                    z = np.clip(z, -3.0, 3.0)
                    data = (z + 3.0) / 6.0
                else:
                    data = np.zeros_like(data)
        else:
            max_val = data.max()
            min_normalize_threshold = 1e-5  # Tiny max usually means almost-black image.
            if max_val > min_normalize_threshold:
                data = data / max_val
            elif max_val > 0:
                # Almost-black image: zero out to avoid amplifying noise.
                data = np.zeros_like(data)
            # else: max_val <= 0, keep as-is (all-zero image).
        
        # Normalize to [-1, 1] if requested (matching original HADSB)
        if self.normalize:
            data = data * 2 - 1
        
        # Add channel dimension
        data = np.expand_dims(data, axis=0)
        
        return data.astype(np.float32)
    
    def __getitem__(self, idx):
        """
        Returns:
            t1: T1 MRI (source) [1, H, W]
            t2: T2 MRI (target) [1, H, W] 
            pet: PET scan (condition) [1, H, W]
            lava_water: (optional) LAVA water [1, H, W]
            lava_fat: (optional) LAVA fat [1, H, W]
            idx: sample index
        """
        base_name = self.base_names[idx]
        
        # Load all three modalities
        t1_path = self.t1_dir / f"{base_name}.npy"
        t2_path = self.t2_dir / f"{base_name}.npy"
        pet_path = self.pet_dir / f"{base_name}.npy"
        
        t1 = self.load_and_preprocess(t1_path, 'mri')
        t2 = self.load_and_preprocess(t2_path, 'mri')
        pet = self.load_and_preprocess(pet_path, 'pet')
        
        # Convert to torch tensors
        t1 = torch.from_numpy(t1)
        t2 = torch.from_numpy(t2)
        pet = torch.from_numpy(pet)
        
        sample = {
            't1': t1,
            't2': t2, 
            'pet': pet,
            'idx': idx,
            'name': base_name
        }
        
        # Load LAVA water if enabled
        if self.use_lava_water and self.lava_water_dir:
            water_path = self.lava_water_dir / f"{base_name}.npy"
            lava_water = self.load_and_preprocess(water_path, 'mri')
            sample['lava_water'] = torch.from_numpy(lava_water)
        
        # Load LAVA fat if enabled
        if self.use_lava_fat and self.lava_fat_dir:
            fat_path = self.lava_fat_dir / f"{base_name}.npy"
            lava_fat = self.load_and_preprocess(fat_path, 'mri')
            sample['lava_fat'] = torch.from_numpy(lava_fat)

        # Body Part semantic conditioning (NEW)
        if self.use_body_part_condition:
            body_part_id = self._get_body_part_id(base_name)
            sample['body_part_id'] = torch.tensor(body_part_id, dtype=torch.long)
            # Also store the string name for debugging
            info = self.region_info.get(base_name, {})
            sample['body_part_name'] = info.get('Body Part', 'Unknown')

            # Also provide organ_mask for organ embedding (from JSON, not CSV)
            organ_mask = self._get_organ_mask(base_name)
            sample['organ_mask'] = torch.tensor(organ_mask, dtype=torch.float32)
            # Store organ names as JSON string to avoid collation issues with variable-length lists
            sample['organ_names'] = json.dumps(info.get('Organs Present', []))

        # Organ cross-attention conditioning (also uses organ_mask)
        elif self.use_organ_crossattn:
            organ_mask = self._get_organ_mask(base_name)
            sample['organ_mask'] = torch.tensor(organ_mask, dtype=torch.float32)
            # Store organ names as JSON string for debugging
            info = self.region_info.get(base_name, {})
            sample['organ_names'] = json.dumps(info.get('Organs Present', []))

        return sample
    
    def get_name(self, idx):
        """Get the name of a sample by index without loading the full sample"""
        return self.base_names[idx]


def build_medical_dataset(opt, log, train=True):
    """Build medical dataset compatible with HADSB training"""
    split = 'train' if train else 'val'

    # LAVA water/fat conditioning
    cond_lava_water = getattr(opt, 'cond_lava_water', False)
    cond_lava_fat = getattr(opt, 'cond_lava_fat', False)
    flat_structure = getattr(opt, 'flat_data_structure', False)
    
    # Body Part / Organ semantic conditioning (NEW)
    cond_body_part = getattr(opt, 'cond_body_part', False)
    cond_organ_crossattn = getattr(opt, 'cond_organ_crossattn', False)
    region_info_file = getattr(opt, 'region_info_file', None)
    
    val_patients = None
    if flat_structure:
        val_patients = getattr(opt, 'val_patients', None)

        # Option 1: load explicit split list from file.
        val_patients_file = getattr(opt, 'val_patients_file', None)
        if val_patients is None and val_patients_file:
            file_path = Path(val_patients_file)
            if file_path.exists():
                val_patients = [line.strip() for line in file_path.read_text().splitlines() if line.strip()]
            else:
                log.warning(f"[Dataset] val_patients_file not found: {val_patients_file}")

        # Option 2: deterministic ratio-based split by group id prefix.
        if val_patients is None:
            val_ratio = float(getattr(opt, 'val_ratio', 0.0))
            if val_ratio > 0:
                t1_dir = Path(opt.data_dir) / 'lava'
                all_t1_files = sorted(glob.glob(str(t1_dir / '*.npy')))
                group_ids = sorted({
                    '_'.join(Path(f).stem.split('_')[:2]) for f in all_t1_files
                })
                if group_ids:
                    split_seed = int(getattr(opt, 'split_seed', 42))
                    rng = np.random.default_rng(split_seed)
                    rng.shuffle(group_ids)
                    n_val = max(1, int(round(len(group_ids) * val_ratio)))
                    val_patients = group_ids[:n_val]
                    log.info(
                        f"[Dataset] Generated validation split from ratio: "
                        f"{n_val}/{len(group_ids)} groups (seed={split_seed})"
                    )
    
    dataset = MedicalModalityDataset(
        data_dir=opt.data_dir,
        split=split,
        image_size=opt.image_size,
        normalize=True,
        pet_log_scale=getattr(opt, 'pet_log_scale', False),
        pet_log_eps=getattr(opt, 'pet_log_eps', 1e-6),
        pet_norm=getattr(opt, 'pet_norm', "max"),
        use_lava_water=cond_lava_water,
        use_lava_fat=cond_lava_fat,
        flat_structure=flat_structure,
        val_patients=val_patients,
        # Body Part / Organ semantic conditioning (JSON-based)
        use_body_part_condition=cond_body_part,
        use_organ_crossattn=cond_organ_crossattn,
        region_info_file=region_info_file,
    )

    log.info(f"[Dataset] Built medical {split} dataset with {len(dataset)} samples")
    if cond_lava_water or cond_lava_fat:
        log.info(
            f"[Dataset] LAVA conditioning enabled (water={cond_lava_water}, fat={cond_lava_fat})"
        )
    if cond_body_part:
        log.info(
            f"[Dataset] Body Part semantic conditioning enabled (region_info={region_info_file})"
        )
    if cond_organ_crossattn:
        log.info(
            f"[Dataset] Organ cross-attention enabled"
        )
    return dataset


def build_medical_dataloader(opt, log, train=True):
    """Build medical dataloader"""
    dataset = build_medical_dataset(opt, log, train)
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size if train else 1,
        shuffle=train,
        num_workers=opt.num_workers if hasattr(opt, 'num_workers') else 4,
        pin_memory=True,
        drop_last=train
    )
    
    return dataloader
