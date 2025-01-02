import sys 
sys.path.append('/home/w4756677/garment/AIpparel-Code/src')
import logging 
log = logging.getLogger(__name__)
from transformers import AutoProcessor, LlavaConfig
from models.aipparel_llava import AIpparelLlavaForConditionalGeneration
import os 
from data.datasets.gcd_dataset import GarmentCodeData
from data.datasets.panel_configs import StandardizeConfig, StatsConfig
from data.garment_tokenizers.garment_tokenizer_for_regression import GarmentTokenizerForRegression
from data.collate_fns import llava_next_collate_fn
from functools import partial
from transformers import LlavaNextProcessor
from models.aipparel_llava_next import AIpparelLlavaNextForConditionalGeneration
from data.garment_tokenizers.special_tokens import PanelEdgeTypeV3

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

train_dataset = GarmentCodeData(
    root_dir="/miele/timur/garmentcodedata",
    editing_dir="/mnt/gs/sci-garment/garmentcodedata_editing",
    caption_dir="/mnt/gs/sci-garment/long-caption-processed",
    editing_flip_prob=0.5,
    sampling_rate=[1,0,0,0,0],
    split_file="/home/w4756677/garment/AIpparel-Code/assets/data_configs/garmentcodedata_datasplit.json",
    datalist_file="assets/data_configs/garmentcodedata_list.txt",
    body_type="default_body",
    panel_classification="assets/data_configs/panel_classes_garmentcodedata.json",
    split="train"
)
val_dataset = GarmentCodeData(
    root_dir="/miele/timur/garmentcodedata",
    editing_dir="/mnt/gs/sci-garment/garmentcodedata_editing",
    caption_dir="/mnt/gs/sci-garment/long-caption-processed",
    editing_flip_prob=0.5,
    sampling_rate=[1,0,0,0,0],
    split_file="/home/w4756677/garment/AIpparel-Code/assets/data_configs/garmentcodedata_datasplit.json",
    datalist_file="assets/data_configs/garmentcodedata_list.txt",
    body_type="default_body",
    panel_classification="assets/data_configs/panel_classes_garmentcodedata.json",
    split="val"
)

gt_stats = StandardizeConfig(
    rotations=StatsConfig(shift=[0, 0, 0, 0], scale=[1, 1, 1, 1]),
    translations=StatsConfig(shift=[-1.25378371e-02,  1.13507532e+02,  2.63046369e+00], scale=[26.06867645, 32.42920198, 22.29905009]),
    vertices=StatsConfig(shift=[8.44428116, 16.84081321], scale=[24.4920733,  26.60402835])
)

garment_tokenizer = GarmentTokenizerForRegression(
    standardize=gt_stats,
    random_tag=True,
    num_tags=108,
    include_template_name=False
)


model = AIpparelLlavaNextForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf"
)
processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

import ipdb; ipdb.set_trace()
processor.tokenizer.pad_token = processor.tokenizer.unk_token
all_new_tokens = garment_tokenizer.get_all_token_names()
num_added_tokens = processor.tokenizer.add_tokens(all_new_tokens, special_tokens=True)
print(f"Added {num_added_tokens} tokens to the tokenizer.")
token_name2_idx_dict = {}
for token in all_new_tokens:
    token_idx = processor.tokenizer(token, add_special_tokens=False).input_ids[0]
    token_name2_idx_dict[token] = token_idx
print(f"Token name to index dictionary: {token_name2_idx_dict}")
garment_tokenizer.set_token_indices(token_name2_idx_dict)

model.config.transf_token = PanelEdgeTypeV3.MOVE.value
model.config.transf_token_index = garment_tokenizer.panel_edge_type_indices.move_idx

model.config.line_token = PanelEdgeTypeV3.LINE.value
model.config.line_token_index = garment_tokenizer.panel_edge_type_indices.line_idx

model.config.quadratic_token = PanelEdgeTypeV3.CURVE.value
model.config.quadratic_token_index = garment_tokenizer.panel_edge_type_indices.curve_idx

model.config.cubic_token = PanelEdgeTypeV3.CUBIC.value
model.config.cubic_token_index = garment_tokenizer.panel_edge_type_indices.cubic_idx

model.config.arc_token = PanelEdgeTypeV3.ARC.value
model.config.arc_token_index = garment_tokenizer.panel_edge_type_indices.arc_idx

model.config.cline_token = PanelEdgeTypeV3.CLOSURE_LINE.value
model.config.cline_token_index = garment_tokenizer.panel_edge_type_indices.closure_line_idx

model.config.cquadratic_token = PanelEdgeTypeV3.CLOSURE_CURVE.value
model.config.cquadratic_token_index = garment_tokenizer.panel_edge_type_indices.closure_curve_idx

model.config.ccubic_token = PanelEdgeTypeV3.CLOSURE_CUBIC.value
model.config.ccubic_token_index = garment_tokenizer.panel_edge_type_indices.closure_cubic_idx

model.config.carc_token = PanelEdgeTypeV3.CLOSURE_ARC.value
model.config.carc_token_index = garment_tokenizer.panel_edge_type_indices.closure_arc_idx
model.resize_token_embeddings(len(processor.tokenizer))

model.enable_input_require_grads()
model.gradient_checkpointing_enable()

collate_fn = partial(llava_next_collate_fn, 
                     processor=processor,
                     garment_tokenizer=garment_tokenizer,
                     model_config="llava-hf/llava-v1.6-mistral-7b-hf")


import ipdb; ipdb.set_trace()
data_dict = collate_fn([train_dataset[0], train_dataset[1]])
print(data_dict)
