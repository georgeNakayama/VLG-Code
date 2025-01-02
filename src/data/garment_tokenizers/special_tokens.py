from enum import Enum
from typing import Dict, List
class ExtendedEnum(Enum):

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))

class DecodeErrorTypes(Enum):
    NO_ERROR = "NO_ERROR"
    UNMATCHED_PATTERN_TOKENS = "UNMATCHED_PATTERN_TOKENS"
    UNMATCHED_PANEL_TOKENS = "UNMATCHED_PANEL_TOKENS"



class SpecialTokensV2(ExtendedEnum):
    PANEL_START = "<panel_start>"
    PANEL_END = "<panel_end>"
    PATTERN_START = "<pattern_start>"
    PATTERN_END = "<pattern_end>"

class PanelEdgeTypeV2(ExtendedEnum):
    MOVE = '<pattern_cmd_MOVE>'
    LINE = '<pattern_cmd_LINE>'
    CLOSURE_LINE = '<pattern_cmd_CLINE>'
    CURVE = '<pattern_cmd_CURVE>'
    CLOSURE_CURVE = '<pattern_cmd_CCURVE>'
    
    def is_closure(self):
        return self in [PanelEdgeTypeV2.CLOSURE_LINE, PanelEdgeTypeV2.CLOSURE_CURVE]
    
    def is_line(self):
        return self in [PanelEdgeTypeV2.LINE, PanelEdgeTypeV2.CLOSURE_LINE]
    
    def is_curve(self):
        return self in [PanelEdgeTypeV2.CURVE, PanelEdgeTypeV2.CLOSURE_CURVE]
    
    def get_num_params(self):
        if self in [PanelEdgeTypeV2.LINE, PanelEdgeTypeV2.CLOSURE_CURVE]:
            return 2
        elif self in [PanelEdgeTypeV2.CURVE]:
            return 4
        elif self == PanelEdgeTypeV2.MOVE:
            return 6
        else:
            return 0


class SpecialTokens(Enum):
    PANEL_START: int = 0
    PANEL_END: int = 1
    PATTERN_START: int = 2
    PATTERN_END: int = 3
    def __str__(self):
        if self == SpecialTokens.PANEL_START:
            return "<panel_start>"
        elif self == SpecialTokens.PANEL_END:
            return "<panel_end>"
        elif self == SpecialTokens.PATTERN_START:
            return "<pattern_start>"
        elif self == SpecialTokens.PATTERN_END:
            return "<pattern_end>"
        
class PanelEdgeTypeV3(ExtendedEnum):
    MOVE = '<pattern_cmd_MOVE>'
    LINE = '<pattern_cmd_LINE>' 
    CLOSURE_LINE = '<pattern_cmd_CLINE>'
    CURVE = '<pattern_cmd_CURVE>'
    CLOSURE_CURVE = '<pattern_cmd_CCURVE>'
    CUBIC = '<pattern_cmd_CUBIC>'
    CLOSURE_CUBIC = '<pattern_cmd_CCUBIC>'
    ARC = '<pattern_cmd_ARC>'
    CLOSURE_ARC = '<pattern_cmd_CARC>'
    
    @classmethod
    def get_sewfactory_token(self):
        return [PanelEdgeTypeV3.MOVE.value, PanelEdgeTypeV3.LINE.value, PanelEdgeTypeV3.CLOSURE_LINE.value, PanelEdgeTypeV3.CURVE.value, PanelEdgeTypeV3.CLOSURE_CURVE.value]
    def is_closure(self):
        return self in [PanelEdgeTypeV3.CLOSURE_LINE, PanelEdgeTypeV3.CLOSURE_CURVE, PanelEdgeTypeV3.CLOSURE_CUBIC, PanelEdgeTypeV3.CLOSURE_ARC]
    
    def is_line(self):
        return self in [PanelEdgeTypeV3.LINE, PanelEdgeTypeV3.CLOSURE_LINE]
    
    def is_curve(self):
        return self in [PanelEdgeTypeV3.CURVE, PanelEdgeTypeV3.CLOSURE_CURVE]
    
    def is_cubic_curve(self):  
        return self in [PanelEdgeTypeV3.CUBIC, PanelEdgeTypeV3.CLOSURE_CUBIC]
    
    def is_arc(self):  
        return self in [PanelEdgeTypeV3.ARC, PanelEdgeTypeV3.CLOSURE_ARC]
    
    def get_num_params(self):
        if self in [PanelEdgeTypeV3.LINE, PanelEdgeTypeV3.CLOSURE_CURVE, PanelEdgeTypeV3.CLOSURE_ARC]:
            return 2
        elif self in [PanelEdgeTypeV3.CURVE, PanelEdgeTypeV3.CLOSURE_CUBIC, PanelEdgeTypeV3.ARC]:
            return 4
        elif self in [PanelEdgeTypeV3.CUBIC]:
            return 6
        elif self == PanelEdgeTypeV3.MOVE:
            return 7
        else:
            return 0
    
    def get_closure(self):
        if self == PanelEdgeTypeV3.LINE:
            return PanelEdgeTypeV3.CLOSURE_LINE
        elif self == PanelEdgeTypeV3.CURVE:
            return PanelEdgeTypeV3.CLOSURE_CURVE
        elif self == PanelEdgeTypeV3.CUBIC:
            return PanelEdgeTypeV3.CLOSURE_CUBIC
        elif self == PanelEdgeTypeV3.ARC:
            return PanelEdgeTypeV3.CLOSURE_ARC
        else:
            return self

class SpecialTokensIndices:
    pattern_start_idx: int
    pattern_end_idx: int
    panel_start_idx: int
    panel_end_idx: int
    
    def __init__(self, token2idx: Dict[str, int]):
        self.pattern_start_idx = token2idx[SpecialTokensV2.PATTERN_START.value]
        self.pattern_end_idx = token2idx[SpecialTokensV2.PATTERN_END.value]
        self.panel_start_idx = token2idx[SpecialTokensV2.PANEL_START.value]
        self.panel_end_idx = token2idx[SpecialTokensV2.PANEL_END.value]
        
    def get_all_indices(self) -> List[int]:
        return [self.pattern_start_idx, self.pattern_end_idx, self.panel_start_idx, self.panel_end_idx]
        
    def get_token_indices(self, token_type: SpecialTokensV2) -> int:
        if token_type == SpecialTokensV2.PATTERN_START:
            return self.pattern_start_idx
        elif token_type == SpecialTokensV2.PATTERN_END:
            return self.pattern_end_idx
        elif token_type == SpecialTokensV2.PANEL_START:
            return self.panel_start_idx
        elif token_type == SpecialTokensV2.PANEL_END:
            return self.panel_end_idx
        
    def get_index_token(self, index: int) -> SpecialTokensV2:
        if index == self.pattern_start_idx:
            return SpecialTokensV2.PATTERN_START
        elif index == self.pattern_end_idx:
            return SpecialTokensV2.PATTERN_END
        elif index == self.panel_start_idx:
            return SpecialTokensV2.PANEL_START
        elif index == self.panel_end_idx:
            return SpecialTokensV2.PANEL_END
        
class PanelEdgeTypeIndices:
    move_idx: int
    line_idx: int
    closure_line_idx: int
    curve_idx: int
    closure_curve_idx: int
    cubic_idx: int
    closure_cubic_idx: int
    arc_idx: int
    closure_arc_idx: int
    
    def __init__(self, token2idx: Dict[str, int], rot_as_quat=False):
        self.move_idx = token2idx.get(PanelEdgeTypeV3.MOVE.value, -1)
        self.line_idx = token2idx.get(PanelEdgeTypeV3.LINE.value, -1)
        self.closure_line_idx = token2idx.get(PanelEdgeTypeV3.CLOSURE_LINE.value, -1)
        self.curve_idx = token2idx.get(PanelEdgeTypeV3.CURVE.value, -1)
        self.closure_curve_idx = token2idx.get(PanelEdgeTypeV3.CLOSURE_CURVE.value, -1)
        self.cubic_idx = token2idx.get(PanelEdgeTypeV3.CUBIC.value, -1)
        self.closure_cubic_idx = token2idx.get(PanelEdgeTypeV3.CLOSURE_CUBIC.value, -1)
        self.arc_idx = token2idx.get(PanelEdgeTypeV3.ARC.value, -1)
        self.closure_arc_idx = token2idx.get(PanelEdgeTypeV3.CLOSURE_ARC.value, -1)
        self.rot_as_quat = rot_as_quat
        
    def get_all_indices(self) -> List[int]:
        return [i for i in [
            self.move_idx, 
            self.line_idx, 
            self.closure_line_idx, 
            self.curve_idx, 
            self.closure_curve_idx, 
            self.cubic_idx, 
            self.closure_cubic_idx, 
            self.arc_idx, 
            self.closure_arc_idx
        ] if i != -1]
        
    def get_all_edge_indices(self) -> List[int]:
        return [i for i in [
            self.line_idx, 
            self.closure_line_idx, 
            self.curve_idx, 
            self.closure_curve_idx, 
            self.cubic_idx, 
            self.closure_cubic_idx, 
            self.arc_idx, 
            self.closure_arc_idx
        ] if i != -1]
    
    def get_token_indices(self, token_type: PanelEdgeTypeV3) -> int:
        if token_type == PanelEdgeTypeV3.MOVE:
            return self.move_idx
        elif token_type == PanelEdgeTypeV3.LINE:
            return self.line_idx
        elif token_type == PanelEdgeTypeV3.CLOSURE_LINE:
            return self.closure_line_idx
        elif token_type == PanelEdgeTypeV3.CURVE:
            return self.curve_idx
        elif token_type == PanelEdgeTypeV3.CLOSURE_CURVE:
            return self.closure_curve_idx
        elif token_type == PanelEdgeTypeV3.CUBIC:
            return self.cubic_idx
        elif token_type == PanelEdgeTypeV3.CLOSURE_CUBIC:
            return self.closure_cubic_idx
        elif token_type == PanelEdgeTypeV3.ARC:
            return self.arc_idx
        elif token_type == PanelEdgeTypeV3.CLOSURE_ARC:
            return self.closure_arc_idx
        
    def get_index_token(self, index: int) -> PanelEdgeTypeV3:
        if index == self.move_idx:
            return PanelEdgeTypeV3.MOVE
        elif index == self.line_idx:
            return PanelEdgeTypeV3.LINE
        elif index == self.closure_line_idx:
            return PanelEdgeTypeV3.CLOSURE_LINE
        elif index == self.curve_idx:
            return PanelEdgeTypeV3.CURVE
        elif index == self.closure_curve_idx:
            return PanelEdgeTypeV3.CLOSURE_CURVE
        elif index == self.cubic_idx:
            return PanelEdgeTypeV3.CUBIC
        elif index == self.closure_cubic_idx:
            return PanelEdgeTypeV3.CLOSURE_CUBIC
        elif index == self.arc_idx:
            return PanelEdgeTypeV3.ARC
        elif index == self.closure_arc_idx:
            return PanelEdgeTypeV3.CLOSURE_ARC
        
    def get_index_param_num(self, index: int) -> int:
        if index == self.move_idx:
            return 6 if not self.rot_as_quat else 7
        return PanelEdgeTypeV3(self.get_index_token(index)).get_num_params()