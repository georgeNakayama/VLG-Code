from typing import Optional
import transformers
import torch
from collections import defaultdict
from models.llava import conversation as conversation_lib
from ..datasets.utils import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX, DEFAULT_PLACEHOLDER_TOKEN, IMAGE_TOKEN_INDEX
from models.llava.mm_utils import tokenizer_image_token, tokenizer_image_and_pattern_token
from data.garment_tokenizers import GarmentTokenizer

def collate_fn_default(
    batch, 
    tokenizer: Optional[transformers.PreTrainedTokenizer]=None, 
    conv_type="llava_v1", 
    use_mm_start_end=True, 
    local_rank=-1
):
    encoded_pattern_list = []
    question_pattern_list = []
    image_path_list = []
    images_clip_list = []
    conversation_list = []
    gt_pattern_list = []
    question_conv_list = []
    questions_list = []
    offset_list = [0]
    cnt = 0
    inferences = []
    for (
        pattern_dict,
        question_pattern,
        image_path, 
        images_clip, 
        conversations, 
        question_only_convs, 
        questions, 
        gt_pattern, 
        inference
        ) in batch:
        # conversation_splits = [con.split(DEFAULT_PLACEHOLDER_TOKEN) for con in conversations]
        # conversation_splits = [[con[0]] + encoded_p + [con[1]] for con, encoded_p in zip(conversation_splits, encoded_pattern)]
        # conversation_list.extend(conversation_splits)
        # conversation_w_splaceholder.extend(conversations)
        encoded_pattern_list.append(pattern_dict['description'])
        question_pattern_list.append(question_pattern)
        # conversation_list.extend(conversations)
        image_path_list.append(image_path)
        images_clip_list.append(images_clip)
        conversation_list.extend(conversations)
        question_conv_list.extend(question_only_convs)
        questions_list.append(questions)
        gt_pattern_list.append(gt_pattern)
        cnt += len(conversations)
        offset_list.append(cnt)
        inferences.append(inference)

    if use_mm_start_end:
        # replace <image> token
        for i in range(len(conversation_list)):
            replace_token = DEFAULT_IMAGE_TOKEN
            replace_token = (
                DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            )
            conversation_list[i] = conversation_list[i].replace(
                DEFAULT_IMAGE_TOKEN, replace_token
            )
        for i in range(len(question_conv_list)):
            replace_token = DEFAULT_IMAGE_TOKEN
            replace_token = (
                DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            )
            question_conv_list[i] = question_conv_list[i].replace(
                DEFAULT_IMAGE_TOKEN, replace_token
            )
    
    pattern_ids = [tokenizer(pattern_tokens, is_split_into_words=True, add_special_tokens=False).input_ids for pattern_tokens in encoded_pattern_list]
    question_pattern_ids = [tokenizer(pattern_tokens, is_split_into_words=True, add_special_tokens=False).input_ids if len(pattern_tokens) > 0 else [] for pattern_tokens in question_pattern_list]
    input_ids = [
        tokenizer_image_and_pattern_token(prompt, tokenizer, pattern_id, pattern_place_holder_token=DEFAULT_PLACEHOLDER_TOKEN, return_tensors="pt")
        for prompt, pattern_id in zip(conversation_list, pattern_ids)
    ]
    questions_ids = [
        tokenizer_image_and_pattern_token(question, tokenizer, question_pattern_id, pattern_place_holder_token=DEFAULT_PLACEHOLDER_TOKEN, return_tensors="pt")
        for question, question_pattern_id in zip(question_conv_list, question_pattern_ids)
    ]
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    questions_ids = torch.nn.utils.rnn.pad_sequence(
        questions_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    attention_masks = input_ids.ne(tokenizer.pad_token_id)
    question_attention_masks = questions_ids.ne(tokenizer.pad_token_id)

    conv = conversation_lib.default_conversation.copy()
    targets = input_ids.clone()

    if conv_type == "llava_v1":
        sep = conv.sep + conv.roles[1] + ": "
    else:
        sep = "[/INST] "
    for conversation, target, pattern_id in zip(conversation_list, targets, pattern_ids):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        n_patterns = 0
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            # if len(parts) != 2:
            #     break
            assert len(parts) == 2, (len(parts), rou)
            parts[0] += sep
            _n_question_patterns = parts[0].count(DEFAULT_PLACEHOLDER_TOKEN)
            instruction_len = len(tokenizer_image_and_pattern_token(parts[0], tokenizer, pattern_id[n_patterns:n_patterns+_n_question_patterns], pattern_place_holder_token=DEFAULT_PLACEHOLDER_TOKEN)) - 2
            _n_patterns = rou.count(DEFAULT_PLACEHOLDER_TOKEN)
            round_len = len(tokenizer_image_and_pattern_token(rou, tokenizer, pattern_id[n_patterns:n_patterns+_n_patterns], pattern_place_holder_token=DEFAULT_PLACEHOLDER_TOKEN))
            n_patterns += _n_patterns


            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if False:
            z = target.clone()
            z = torch.where(z == IGNORE_INDEX, tokenizer.unk_token_id, z)
            if local_rank == 0:
                print(
                    "conversation: ",
                    conversation,
                    "tokenizer.decode(z): ",
                    tokenizer.decode(z),
                )

        if cur_len < tokenizer.model_max_length:
            if not cur_len == total_len and not n_patterns == len(pattern_id):
                print(conversation_list)
                print("cur_len: ", cur_len, "total_len: ", total_len)
                print(target)
                print(input_ids)
                print(image_path_list)
                print(tokenizer.decode(torch.where(input_ids[0] == IGNORE_INDEX, tokenizer.unk_token_id, input_ids[0])))
                exit()
    input_len = input_ids.shape[1] 
    if inferences[0] == False:
        truncate_len = tokenizer.model_max_length - 255

        if input_ids.shape[1] > truncate_len:
            input_ids = input_ids[:, :truncate_len]
            targets = targets[:, :truncate_len]
            attention_masks = attention_masks[:, :truncate_len]

    return {
        "input_len": input_len,
        "image_paths": image_path_list,
        "images_clip": torch.stack(images_clip_list, dim=0),
        "input_ids": input_ids,
        "labels": targets,
        "gt_patterns": gt_pattern_list,
        "attention_masks": attention_masks,
        "offset": torch.LongTensor(offset_list),
        "questions_list": questions_list,
        "question_ids": questions_ids,
        "question_attention_masks": question_attention_masks,
        "inference": inferences[0],
        "conversation_list": conversation_list,
    }

# check system prompt token seq or user prompt token seq is in the current token list
def find_header(targets, seq):
    assert isinstance(targets[0], int)
    n = len(targets)
    sep_inds = []
    for i in range(len(seq)-n):
        if seq[i:i+n] in targets:
            sep_inds.append(i)
    return sep_inds

def llava_next_collate_fn(
    batch, 
    processor: Optional[transformers.AutoProcessor]=None, 
    garment_tokenizer: Optional[GarmentTokenizer]=None, 
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
        n_question_pattern = dialog.count(DEFAULT_PLACEHOLDER_TOKEN)
        n_pattern = dialog.count(DEFAULT_PLACEHOLDER_TOKEN)
        assert n_pattern == n_question_pattern + 1, "only support one pattern output for now"
        pattern_list.append(out_patterns)
        pattern_dicts = garment_tokenizer.encode_list(out_patterns)
        question_endpoint_cnt = 0
        question_transf_cnt = 0
        assert len(pattern_dicts) == n_pattern
        for i in range(n_pattern):
            pattern_dict = pattern_dicts[i]
            dialog = dialog.replace(DEFAULT_PLACEHOLDER_TOKEN, " ".join(pattern_dict['description']), 1)
            concated_endpoints = torch.cat(pattern_dict['endpoints'])
            pattern_endpoints_list.append(concated_endpoints)
            question_endpoint_cnt += concated_endpoints.shape[0] - pattern_dict['endpoints'][-1].shape[0]
            concated_transf = torch.cat(pattern_dict['transformations'])
            pattern_transf_list.append(concated_transf)
            question_transf_cnt += concated_transf.shape[0] - pattern_dict['transformations'][-1].shape[0]
            if i < n_question_pattern:
                question = question.replace(DEFAULT_PLACEHOLDER_TOKEN, " ".join(pattern_dict['description']), 1)
            else:
                # only keep the output pattern's params
                pattern_param_list.append(pattern_dict['params'])
        dialog_list.append(dialog)
        question_list.append(question)  
        question_endpoint_cnt_list.append(question_endpoint_cnt)
        question_transf_cnt_list.append(question_transf_cnt)
    
    questions_batch = processor(images=image_list, text=question_list, return_tensors="pt", padding=True)
    input_batch = processor(images=image_list, text=dialog_list, return_tensors="pt", padding=True)
    
    pattern_endpoints = torch.nn.utils.rnn.pad_sequence(pattern_endpoints_list, batch_first=True, padding_value=0)
    pattern_transfs = torch.nn.utils.rnn.pad_sequence(pattern_transf_list, batch_first=True, padding_value=0)
    pattern_endpoint_masks = torch.arange(pattern_endpoints.shape[1]).unsqueeze(0) < torch.tensor([pattern_endpoints.shape[0] for pattern_endpoints in pattern_endpoints_list]).unsqueeze(1)
    pattern_transf_masks = torch.arange(pattern_transfs.shape[1]).unsqueeze(0) < torch.tensor([pattern_transf.shape[0] for pattern_transf in pattern_transf_list]).unsqueeze(1)
    question_pattern_endpoints_mask = torch.arange(pattern_endpoints.shape[1]).unsqueeze(0) < torch.tensor(quesetion_endpoint_cnt_list).unsqueeze(1)
    question_pattern_transfs_mask = torch.arange(pattern_transfs.shape[1]).unsqueeze(0) < torch.tensor(question_transf_cnt_list).unsqueeze(1)
    
    label_list = []
    inst_header = [733, 28748, 16289, 28793]
    sep_header = [16910, 28713, 28767]
    for i in range(len(input_batch["input_ids"])):
        dialog_tokens = batch["input_ids"][i].tolist()
        labels = copy.copy(dialog_tokens)
        last_idx = 0
        sep_inds = find_header(sep_header,labels)
        for n, idx in enumerate(sep_inds):
            current_seq = labels[last_idx:idx+3] # since len(sep_header) == 3
            inst_end_inds = find_header(inst_header, current_seq)
            assert len(inst_end_inds) == 1, "only support one instruction for now"
            labels[last_idx:inst_end_inds[0] + 4 + last_idx] = [IGNORE_INDEX] * (inst_end_inds[0]+4-last_idx)
            last_idx = idx+3
        labels[last_idx:] = [IGNORE_INDEX] * (len(labels) - last_idx) # mask padding
        label_list.append(labels)

    pattern_param_keys = set([k for pattern_param in pattern_param_list for k in pattern_param.keys()])
    pattern_params = dict()
    pattern_params_mask = dict()
    for key in pattern_param_keys:
        padded_params = torch.nn.utils.rnn.pad_sequence([v.transpose(0, 1) for pattern_param in pattern_param_list for k, v in pattern_param.items() if k == key], batch_first=True, padding_value=0)
        pattern_params[key] = padded_params.transpose(1, 2)
        pattern_params_mask[key] = torch.arange(padded_params.shape[1]).unsqueeze(0) < torch.tensor([padded_param[key].shape[0] if key in padded_param else 0 for padded_param in pattern_param_list]).unsqueeze(1)

    input_len = input_ids.shape[1] 
    return_dict = {
        "input_len": input_len,
        "image_paths": image_path_list,
        "sample_type": torch.LongTensor(sample_type_list),
        "labels": torch.tensor(label_list),
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
        "question_ids": questions_ids,
    }
    return_dict.update(input_batch)
    return return_dict
    
    
def collate_fn_regression_dresscode(
    batch, 
    pad_id=0,
    local_rank=-1
):
    input_id_list = []
    param_targets = []
    pattern_endpoints_list = []
    pattern_transf_list = []
    gt_pattern_list = []
    caption_feature_list = []
    caption_list = []
    name_list = []
    for (
        data_name, 
        caption,
        pattern_dict,
        caption_features, 
        gt_pattern, 
        ) in batch:
        caption_list.append(caption)
        name_list.append(data_name)
        input_id_list.append(pattern_dict['description'])
        param_targets.append(pattern_dict['params'])
        pattern_endpoints_list.append(pattern_dict['endpoints'])
        pattern_transf_list.append(pattern_dict['transformations'])
        gt_pattern_list.append(gt_pattern)
        caption_feature_list.append(caption_features)

    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_id_list, batch_first=True, padding_value=pad_id
    )
    targets = input_ids.clone()
    targets[targets == pad_id] = -100
    endpoint_max_len = max([len(endpoints) for endpoints in pattern_endpoints_list])
    pattern_endpoints = torch.zeros(len(pattern_endpoints_list), endpoint_max_len, 2, dtype=torch.float32)
    pattern_endpoint_masks = torch.zeros(len(pattern_endpoints_list), endpoint_max_len, dtype=torch.bool)
    for i, endpoints in enumerate(pattern_endpoints_list):
        if len(endpoints) == 0:
            continue
        pattern_endpoints[i, :len(endpoints)] = endpoints
        pattern_endpoint_masks[i, :len(endpoints)] = True
    pattern_transf_max_len = max([len(transf) for transf in pattern_transf_list])
    pattern_transfs = torch.zeros(len(pattern_transf_list), pattern_transf_max_len, pattern_transf_list[0].shape[-1], dtype=torch.float32)
    pattern_transf_masks = torch.zeros(len(pattern_transf_list), pattern_transf_max_len, dtype=torch.bool)
    for i, transf in enumerate(pattern_transf_list):
        if len(transf) == 0:
            continue
        pattern_transfs[i, :len(transf)] = transf
        pattern_transf_masks[i, :len(transf)] = True
        
    param_targets_keys = set([k for param_target in param_targets for k in param_target.keys()])
    new_param_targets = dict()
    param_target_masks = dict()
    for key in param_targets_keys:
        max_len = max([len(param_target[key]) for param_target in param_targets if key in param_target])
        shape = (len(param_targets), max_len, [param_target[key].shape[-1] for param_target in param_targets if key in param_target][0])
        new_param_targets[key] = torch.zeros(*shape, dtype=torch.float32)
        param_target_masks[key] = torch.zeros(len(param_targets), max_len, dtype=torch.bool)
        for i, param_target in enumerate(param_targets):
            if key in param_target:
                param_target_masks[key][i, :len(param_target[key])] = True
                new_param_targets[key][i, :len(param_target[key])] = param_target[key]
    return {
        "caption_features": torch.stack(caption_feature_list, dim=0),
        "input_ids": input_ids,
        "labels": targets,
        "param_targets": new_param_targets,
        "param_target_endpoints": pattern_endpoints,
        "param_target_endpoints_mask": pattern_endpoint_masks,
        "param_target_transformations": pattern_transfs,
        "param_target_transformations_mask": pattern_transf_masks,
        "param_target_masks": param_target_masks,
        "gt_patterns": gt_pattern_list,
        "data_names": name_list,
        "captions": caption_list,
    }
    
def collate_fn_dresscode(
    batch, 
    pad_id=0,
    local_rank=-1
):
    input_id_list = []
    gt_pattern_list = []
    caption_feature_list = []
    caption_list = []
    name_list = []
    for (
        data_name, 
        caption,
        pattern_dict,
        caption_features, 
        gt_pattern, 
        ) in batch:
        caption_list.append(caption)
        name_list.append(data_name)
        input_id_list.append(pattern_dict['description'])
        gt_pattern_list.append(gt_pattern)
        caption_feature_list.append(caption_features)
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_id_list, batch_first=True, padding_value=pad_id
    )
    targets = input_ids.clone()
    targets[targets == pad_id] = -100
    return {
        "caption_features": torch.stack(caption_feature_list, dim=0),
        "input_ids": input_ids,
        "labels": targets,
        "gt_patterns": gt_pattern_list,
        "data_names": name_list,
        "captions": caption_list,
    }