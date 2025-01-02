from typing import Optional, Literal
import transformers
import torch
import copy
from .datasets.utils import IGNORE_INDEX, DEFAULT_PLACEHOLDER_TOKEN
from transformers import AutoProcessor
from data.garment_tokenizers.default_garment_tokenizer import GarmentTokenizer

# check system prompt token seq or user prompt token seq is in the current token list
def find_header(targets, seq):
    n = targets.shape[1]
    sep_inds = []
    seq = seq[None]
    for i in range(seq.shape[1]-n+1):
        if torch.all(seq[:, i:i+n] == targets, axis=-1).any():
            sep_inds.append(i)
    return sep_inds

def collate_fn(
    batch, 
    processor: Optional[transformers.AutoProcessor]=None, 
    garment_tokenizer: Optional[GarmentTokenizer]=None, 
    model_version: Literal["llava-hf/llava-v1.6-mistral-7b-hf", "meta-llama/Llama-3.2-11B-Vision-Instruct"]="llava-hf/llava-v1.6-mistral-7b-hf"
):
    sample_type_list = []
    image_path_list = []
    images_list = []
    dialog_list = []
    question_list = []
    pattern_list = []
    pattern_endpoints_list = []
    pattern_transf_list = []
    pattern_param_list = []
    question_endpoint_cnt_list = []
    question_transf_cnt_list = []
    for image_path, image, dialog, question, out_patterns, sample_type in batch:
        sample_type_list.append(sample_type)
        image_path_list.append(image_path)
        images_list.append(image)
        dialog = processor.apply_chat_template(dialog, tokenize=False)
        question = processor.apply_chat_template(question, tokenize=False)
        n_question_pattern = question.count(DEFAULT_PLACEHOLDER_TOKEN)
        n_pattern = dialog.count(DEFAULT_PLACEHOLDER_TOKEN)
        assert n_pattern - n_question_pattern in [0, 1], "only support one or zero pattern output for now"
        pattern_list.append(out_patterns)
        pattern_dicts = garment_tokenizer.encode_list(out_patterns)
        question_endpoint_cnt = 0
        _endpoint_list = []
        question_transf_cnt = 0
        _transf_list = []
        assert len(pattern_dicts) == n_pattern
        for i in range(n_pattern):
            pattern_dict = pattern_dicts[i]
            dialog = dialog.replace(DEFAULT_PLACEHOLDER_TOKEN, "".join(pattern_dict['description']), 1)
            _endpoint_list.append(pattern_dict['endpoints'])
            _transf_list.append(pattern_dict['transformations'])
            if i < n_question_pattern:
                question_endpoint_cnt += pattern_dict['endpoints'].shape[0]
                question_transf_cnt += pattern_dict['transformations'].shape[0]
                question = question.replace(DEFAULT_PLACEHOLDER_TOKEN, " ".join(pattern_dict['description']), 1)
            else:
                # only keep the output pattern's params
                pattern_param_list.append(pattern_dict['params'])
        dialog_list.append(dialog)
        question_list.append(question)  
        question_endpoint_cnt_list.append(question_endpoint_cnt)
        question_transf_cnt_list.append(question_transf_cnt)
        pattern_endpoints_list.append(torch.cat(_endpoint_list))
        pattern_transf_list.append(torch.cat(_transf_list))
    
    questions_batch = processor(images=images_list, text=question_list, return_tensors="pt", padding=True)
    input_batch = processor(images=images_list, text=dialog_list, return_tensors="pt", padding=True)
    
    pattern_endpoints = torch.nn.utils.rnn.pad_sequence(pattern_endpoints_list, batch_first=True, padding_value=0)
    pattern_transfs = torch.nn.utils.rnn.pad_sequence(pattern_transf_list, batch_first=True, padding_value=0)
    pattern_endpoint_masks = torch.arange(pattern_endpoints.shape[1]).unsqueeze(0) < torch.tensor([pattern_endpoints.shape[0] for pattern_endpoints in pattern_endpoints_list]).unsqueeze(1)
    pattern_transf_masks = torch.arange(pattern_transfs.shape[1]).unsqueeze(0) < torch.tensor([pattern_transf.shape[0] for pattern_transf in pattern_transf_list]).unsqueeze(1)
    question_pattern_endpoints_mask = torch.arange(pattern_endpoints.shape[1]).unsqueeze(0) < torch.tensor(question_endpoint_cnt_list).unsqueeze(1)
    question_pattern_transfs_mask = torch.arange(pattern_transfs.shape[1]).unsqueeze(0) < torch.tensor(question_transf_cnt_list).unsqueeze(1)
    
    labels = construct_labels(input_batch["input_ids"], processor, model_version)

    pattern_param_keys = set([k for pattern_param in pattern_param_list for k in pattern_param.keys()])
    pattern_params = dict()
    pattern_params_mask = dict()
    for key in pattern_param_keys:
        padded_params = torch.nn.utils.rnn.pad_sequence([v for pattern_param in pattern_param_list for k, v in pattern_param.items() if k == key], batch_first=True, padding_value=0)
        pattern_params[key] = padded_params
        pattern_params_mask[key] = torch.arange(padded_params.shape[1]).unsqueeze(0) < torch.tensor([padded_param[key].shape[0] if key in padded_param else 0 for padded_param in pattern_param_list]).unsqueeze(1)

    input_len = input_batch["input_ids"].shape[1] 
    return_dict = {
        "input_len": input_len,
        "image_paths": image_path_list,
        "sample_type": torch.LongTensor(sample_type_list),
        "labels": labels,
        "pattern_params": pattern_params,
        "pattern_params_mask": pattern_params_mask,
        "pattern_endpoints": pattern_endpoints,
        "pattern_endpoint_masks": pattern_endpoint_masks,
        "pattern_transfs": pattern_transfs,
        "pattern_transf_masks": pattern_transf_masks,
        "gt_patterns": pattern_list,
        "questions_list": question_list,
        "question_pattern_endpoints_mask": question_pattern_endpoints_mask,
        "question_pattern_transfs_mask": question_pattern_transfs_mask,
    }
    return_dict.update(input_batch)
    return return_dict

def construct_labels(
    input_ids: torch.LongTensor, 
    processor: AutoProcessor,
    model_version: Literal[
        "llava-hf/llava-v1.6-mistral-7b-hf", 
        "meta-llama/Llama-3.2-11B-Vision-Instruct"
    ]
):
    label_list = []
    if model_version == "llava-hf/llava-v1.6-mistral-7b-hf":
        image_token_id = 32000
        for i in range(len(input_ids)):
            dialog_tokens = input_ids[i]
            labels = dialog_tokens.clone()
            last_idx = 0
            inst_header = torch.tensor([[733, 28748, 16289, 28793]]) # [/INST] to [733, 28748, 16289, 28793]
            sep_header = torch.tensor([[16910, 28713, 28767, 28705]]) # </s> to [16910, 28713, 28767, 28705]
            sep_inds = find_header(sep_header,labels)
            for n, idx in enumerate(sep_inds):
                current_seq = labels[last_idx:idx+3]
                inst_end_inds = find_header(inst_header, current_seq)
                assert len(inst_end_inds) == 1, "only support one instruction for now"
                labels[last_idx:inst_end_inds[0] + 4 + last_idx] = IGNORE_INDEX
                last_idx = idx+4
    elif model_version == "meta-llama/Llama-3.2-11B-Vision-Instruct":
        image_token_id = 128256
        prompt_header_seqs = torch.tensor([[128006, 9125, 128007],[128006, 882, 128007]])
        assistant_header_seq = torch.tensor([[128006, 78191, 128007]])
        for i in range(len(input_ids)):
            dialog_tokens = input_ids[i]
            labels = dialog_tokens.clone()
            last_idx = 0
            eot_indices = (input_ids[i] == 128009).nonzero().flatten()
            # system prompt header "<|start_header_id|>system<|end_header_id|>" has been tokenized to [128006, 9125, 128007]
            # user prompt header "<|start_header_id|>user<|end_header_id|>" has been tokenized to [128006, 882, 128007]
            for n, idx in enumerate(eot_indices):
                current_seq = labels[last_idx:idx+1]
                if len(find_header(prompt_header_seqs,current_seq)) > 0:
                    # found prompt header, indicating that this seq should be masked
                    labels[last_idx:idx+1] = IGNORE_INDEX
                else:
                    last_idx = idx+1
            #  Mask all the assistant header prompt <|start_header_id|>assistant<|end_header_id|>, which has been tokenized to [128006, 78191, 128007]
            header_ind = find_header(assistant_header_seq,labels)[0]
            labels[header_ind: header_ind + 3] = IGNORE_INDEX
        
    label_list.append(labels)
    labels = torch.stack(label_list)
    # Mask the padding token and image token
    pad_or_image_mask = torch.isin(labels, torch.tensor([processor.tokenizer.pad_token_id, image_token_id]))
    labels[pad_or_image_mask] = IGNORE_INDEX
    return labels
    
    
    
    