"""
Region and Organ Configuration for Medical Image Analysis

This module defines the body parts and organs used in the medical imaging pipeline.
Names are stored as strings for semantic embedding using PubMedBERT.
"""

from typing import List, Dict

# ============================================================================
# Body Parts (11 classes)
# Represents anatomical regions in axial MRI slices
# ============================================================================
BODY_PARTS: List[str] = [
    "Upper Chest",      # 0: Lung apex, clavicle area
    "Middle Chest",     # 1: Heart level, mid-thorax
    "Lower Chest",      # 2: Diaphragm level, lower thorax
    "Upper Abdomen",    # 3: Liver, spleen, stomach
    "Middle Abdomen",   # 4: Kidneys, pancreas
    "Lower Abdomen",    # 5: Intestines, lower kidneys
    "Pelvis",           # 6: Pelvic organs, sacrum
    "Hips",             # 7: Hip joints, femoral heads
    "Thighs",           # 8: Upper legs, femur
    "Neck",             # 9: Cervical region
    "Face & Jaw",       # 10: Facial structures
]

# ============================================================================
# Organs (36 classes)
# Represents anatomical structures visible in MRI
# Ordered by frequency in dataset (most common first)
# ============================================================================
ORGANS: List[str] = [
    "Subcutaneous Fat",     # 0: Present in almost all slices
    "Ribs",                 # 1: Thoracic/upper abdominal
    "Colon",                # 2: Abdominal/pelvic
    "Psoas",                # 3: Psoas muscle
    "Thoracic Spine",       # 4: Thoracic vertebrae
    "Gluteus",              # 5: Gluteal muscles
    "Abdominal Aorta",      # 6: Major vessel
    "Pelvic Bone",          # 7: Pelvic structures
    "Lumbar Spine",         # 8: Lumbar vertebrae
    "Pectoralis Muscle",    # 9: Chest muscle
    "Liver",                # 10: Largest solid organ
    "Spleen",               # 11: Left upper quadrant
    "Rectum",               # 12: Pelvic organ
    "Stomach",              # 13: Upper GI
    "Descending Aorta",     # 14: Thoracic vessel
    "Cervical Spine",       # 15: Neck vertebrae
    "L-Kidney",             # 16: Left kidney
    "Quadriceps",           # 17: Thigh muscle
    "Lower Lungs",          # 18: Lung bases
    "R-Kidney",             # 19: Right kidney
    "Femur",                # 20: Thigh bone
    "Lung Apex",            # 21: Upper lung
    "Sacrum",               # 22: Sacral bone
    "Femoral Head",         # 23: Hip joint
    "Hip Joint",            # 24: Acetabulum
    "Clavicle",             # 25: Collar bone
    "Heart",                # 26: Cardiac
    "Uterus/Prostate",      # 27: Pelvic organ
    "Aortic Arch",          # 28: Major vessel
    "Bladder",              # 29: Pelvic organ
    "Diaphragm",            # 30: Respiratory muscle
    "Tongue",               # 31: Oral structure
    "Mandible",             # 32: Jaw bone
    "Upper Liver",          # 33: Superior liver
    "Maxilla",              # 34: Upper jaw
    "Rectant",              # 35: (possibly typo in original data)
]

# ============================================================================
# Constants
# ============================================================================
NUM_BODY_PARTS: int = len(BODY_PARTS)  # 11
NUM_ORGANS: int = len(ORGANS)  # 36

# ============================================================================
# Mapping dictionaries (auto-generated from lists)
# ============================================================================
BODY_PART_TO_IDX: Dict[str, int] = {bp: i for i, bp in enumerate(BODY_PARTS)}
IDX_TO_BODY_PART: Dict[int, str] = {i: bp for i, bp in enumerate(BODY_PARTS)}

ORGAN_TO_IDX: Dict[str, int] = {organ: i for i, organ in enumerate(ORGANS)}
IDX_TO_ORGAN: Dict[int, str] = {i: organ for i, organ in enumerate(ORGANS)}

# ============================================================================
# Semantic Embedding Configuration
# ============================================================================
SEMANTIC_EMBEDDING_CONFIG: Dict = {
    # PubMedBERT for medical domain knowledge
    "model_name": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
    
    # Whether to freeze the encoder (recommended for efficiency)
    "freeze_encoder": True,
    
    # Prompt templates for better semantic understanding
    "body_part_prompt": "MRI axial slice of the {name} region",
    "organ_prompt": "MRI showing the {name} anatomical structure",
    
    # Output dimensions
    "hidden_dim": 768,  # PubMedBERT output dimension
    "body_part_output_dim": 64,
    "organ_output_dim": 64,
}

# ============================================================================
# Anatomical groupings (for potential hierarchical modeling)
# ============================================================================
BODY_PART_GROUPS: Dict[str, List[str]] = {
    "Thorax": ["Upper Chest", "Middle Chest", "Lower Chest"],
    "Abdomen": ["Upper Abdomen", "Middle Abdomen", "Lower Abdomen"],
    "Pelvis": ["Pelvis", "Hips"],
    "Extremities": ["Thighs"],
    "Head & Neck": ["Neck", "Face & Jaw"],
}

ORGAN_CATEGORIES: Dict[str, List[str]] = {
    "Solid Organs": ["Liver", "Spleen", "L-Kidney", "R-Kidney", "Heart"],
    "Hollow Organs": ["Stomach", "Colon", "Bladder", "Rectum"],
    "Vessels": ["Abdominal Aorta", "Descending Aorta", "Aortic Arch"],
    "Bones": ["Ribs", "Thoracic Spine", "Lumbar Spine", "Cervical Spine", 
              "Pelvic Bone", "Sacrum", "Femur", "Femoral Head", "Clavicle",
              "Mandible", "Maxilla"],
    "Muscles": ["Psoas", "Gluteus", "Pectoralis Muscle", "Quadriceps", "Diaphragm"],
    "Fat": ["Subcutaneous Fat"],
    "Lungs": ["Lower Lungs", "Lung Apex"],
    "Reproductive": ["Uterus/Prostate"],
    "Other": ["Hip Joint", "Tongue", "Upper Liver", "Rectant"],
}


def get_body_part_idx(name: str) -> int:
    """Get index for a body part name, returns -1 if not found."""
    return BODY_PART_TO_IDX.get(name, -1)


def get_organ_idx(name: str) -> int:
    """Get index for an organ name, returns -1 if not found."""
    return ORGAN_TO_IDX.get(name, -1)


def create_organ_mask(organ_names: List[str]) -> List[int]:
    """Create a multi-hot mask for a list of organ names."""
    mask = [0] * NUM_ORGANS
    for name in organ_names:
        idx = get_organ_idx(name)
        if idx >= 0:
            mask[idx] = 1
    return mask

