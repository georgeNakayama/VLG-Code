from dataclasses import dataclass
from typing import Literal, Optional

@dataclass 
class DataSplitConfig:
    type: Literal["percent", "count"] = "percent"
    split_on: Literal["pcd", "folder"] = "folder"
    valid_per_type: int = 5 
    test_per_type: int = 5
    load_from_split_file: Optional[str] = None