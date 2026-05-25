"""
Backwards compatible shell for dataset.
"""

from src.data.dataset import CocoQADataset, build_cocoqa_datasets, TYPE_TO_ID, ID_TO_TYPE
from src.data.transforms import default_image_transform
from src.data.helpers import read_json, read_jsonl