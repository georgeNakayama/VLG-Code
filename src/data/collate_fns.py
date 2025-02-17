from typing import Literal

import torch
import transformers
from transformers import AutoProcessor

from .datasets.utils import IGNORE_INDEX, DEFAULT_PLACEHOLDER_TOKEN, LLAMA_EOT_INDEX, LLAMA_IMAGE_INDEX
from data.garment_tokenizers.default_garment_tokenizer import GarmentTokenizer


# check system prompt token seq or user prompt token seq is in the current token list
def find_header(targets, seq):
    n = targets.shape[1]
    sep_inds = []
    seq = seq[None]
    for i in range(seq.shape[1] - n + 1):
        if torch.all(seq[:, i : i + n] == targets, axis=-1).any():
            sep_inds.append(i)
    return sep_inds


def collate_fn(
    batch,
    processor: transformers.AutoProcessor,
    garment_tokenizer: GarmentTokenizer,
    model_version: Literal[
        "llava-hf/llava-v1.6-mistral-7b-hf", "meta-llama/Llama-3.2-11B-Vision-Instruct"
    ] = "llava-hf/llava-v1.6-mistral-7b-hf",
    inference: bool = False,
):
    # Outputs of get function of dataset
    image_path_list = []
    image_list = []
    dialog_list = []
    prompt_list = []
    pattern_list = []
    sample_type_list = []

    # Additional data from sewing pattern
    pattern_endpoints_list = []
    pattern_transf_list = []
    pattern_param_list = []
    question_endpoint_cnt_list = []
    question_transf_cnt_list = []

    for image_path, image, dialog, prompt, out_patterns, sample_type in batch:
        image_path_list.append(image_path)
        image_list.append(image)
        pattern_list.append(out_patterns)
        sample_type_list.append(sample_type)

        dialog = processor.apply_chat_template(
            dialog, tokenize=False, add_generation_prompt=False
        )
        prompt = processor.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )

        n_dialog_pattern = dialog.count(DEFAULT_PLACEHOLDER_TOKEN)
        n_prompt_pattern = prompt.count(DEFAULT_PLACEHOLDER_TOKEN)
        assert n_dialog_pattern - n_prompt_pattern in [
            0,
            1,
        ], "Only support one or zero pattern output!"

        pattern_dicts = garment_tokenizer.encode_list(out_patterns)
        assert len(pattern_dicts) == n_dialog_pattern

        endpoint_list = []
        transf_list = []
        question_endpoint_count = 0
        question_transf_count = 0

        for i in range(n_dialog_pattern):
            pattern_dict = pattern_dicts[i]
            dialog = dialog.replace(
                DEFAULT_PLACEHOLDER_TOKEN, "".join(pattern_dict["description"]), 1
            )
            endpoint_list.append(pattern_dict["endpoints"])
            transf_list.append(pattern_dict["transformations"])
            if i < n_prompt_pattern:
                question_endpoint_count += pattern_dict["endpoints"].shape[0]
                question_transf_count += pattern_dict["transformations"].shape[0]
                prompt = prompt.replace(
                    DEFAULT_PLACEHOLDER_TOKEN, " ".join(pattern_dict["description"]), 1
                )
            else:
                # only keep the output pattern's params
                pattern_param_list.append(pattern_dict["params"])

        dialog_list.append(dialog)
        prompt_list.append(prompt)
        question_endpoint_cnt_list.append(question_endpoint_count)
        question_transf_cnt_list.append(question_transf_count)
        pattern_endpoints_list.append(torch.cat(endpoint_list))
        pattern_transf_list.append(torch.cat(transf_list))

    # Adjust for different types
    pattern_endpoints = torch.nn.utils.rnn.pad_sequence(
        pattern_endpoints_list, batch_first=True, padding_value=0
    )
    pattern_transfs = torch.nn.utils.rnn.pad_sequence(
        pattern_transf_list, batch_first=True, padding_value=0
    )

    # TODO(jan): Check this shit
    pattern_endpoint_masks = torch.arange(pattern_endpoints.shape[1]).unsqueeze(
        0
    ) < torch.tensor(
        [pattern_endpoints.shape[0] for pattern_endpoints in pattern_endpoints_list]
    ).unsqueeze(
        1
    )
    pattern_transf_masks = torch.arange(pattern_transfs.shape[1]).unsqueeze(
        0
    ) < torch.tensor(
        [pattern_transf.shape[0] for pattern_transf in pattern_transf_list]
    ).unsqueeze(
        1
    )
    question_pattern_endpoints_mask = torch.arange(
        pattern_endpoints.shape[1]
    ).unsqueeze(0) < torch.tensor(question_endpoint_cnt_list).unsqueeze(1)
    question_pattern_transfs_mask = torch.arange(pattern_transfs.shape[1]).unsqueeze(
        0
    ) < torch.tensor(question_transf_cnt_list).unsqueeze(1)

    # Convert the inputs to tensors
    inference_batch = processor(
        images=image_list, text=prompt_list, return_tensors="pt", padding=True
    )
    train_batch = processor(
        images=image_list, text=dialog_list, return_tensors="pt", padding=True
    )

    labels = construct_labels(train_batch["input_ids"], processor, model_version)

    # TODO(jan): Check this as well
    pattern_param_keys = set(
        [k for pattern_param in pattern_param_list for k in pattern_param.keys()]
    )
    pattern_params = dict()
    pattern_params_mask = dict()
    for key in pattern_param_keys:
        param_num = -1
        for pattern_param in pattern_param_list:
            if key in pattern_param:
                param_num = pattern_param[key].shape[-1]
                break
        padded_params = torch.nn.utils.rnn.pad_sequence(
            [
                (
                    pattern_param[key]
                    if key in pattern_param
                    else torch.zeros(1, param_num)
                )
                for pattern_param in pattern_param_list
            ],
            batch_first=True,
            padding_value=0,
        )
        pattern_params[key] = padded_params
        pattern_params_mask[key] = torch.arange(padded_params.shape[1]).unsqueeze(
            0
        ) < torch.tensor(
            [
                padded_param[key].shape[0] if key in padded_param else 0
                for padded_param in pattern_param_list
            ]
        ).unsqueeze(
            1
        )

    # Construct the inputs for the model depending on whether we sample or not
    return_dict = {
        "sample_type": torch.LongTensor(sample_type_list),
        "pattern_params": pattern_params,
        "pattern_params_mask": pattern_params_mask,
        "pattern_endpoints": pattern_endpoints,
        "pattern_transfs": pattern_transfs,
    }
    if inference:
        return_dict.update(inference_batch)
        return_dict.update(
            {
                "image_paths": image_path_list,
                "gt_patterns": pattern_list,
                "questions_list": prompt_list,
                "pattern_endpoint_masks": question_pattern_endpoints_mask,
                "pattern_transf_masks": question_pattern_transfs_mask,
                "gt_ids": train_batch["input_ids"],
            }
        )
    else:
        return_dict.update(train_batch)
        return_dict.update(
            {
                "labels": labels,
                "pattern_endpoint_masks": pattern_endpoint_masks,
                "pattern_transf_masks": pattern_transf_masks,
            }
        )
    return return_dict


def construct_labels(
    input_ids: torch.LongTensor,
    processor: AutoProcessor,
    model_version: Literal[
        "llava-hf/llava-v1.6-mistral-7b-hf", "meta-llama/Llama-3.2-11B-Vision-Instruct"
    ],
):
    label_list = []
    if model_version == "llava-hf/llava-v1.6-mistral-7b-hf":
        image_token_id = 32000
        for i in range(len(input_ids)):
            dialog_tokens = input_ids[i]
            labels = dialog_tokens.clone()
            last_idx = 0
            inst_header = torch.tensor(
                [[733, 28748, 16289, 28793]]
            )  # [/INST] to [733, 28748, 16289, 28793]
            sep_header = torch.tensor(
                [[16910, 28713, 28767, 28705]]
            )  # </s> to [16910, 28713, 28767, 28705]
            sep_inds = find_header(sep_header, labels)
            for n, idx in enumerate(sep_inds):
                current_seq = labels[last_idx : idx + 3]
                inst_end_inds = find_header(inst_header, current_seq)
                assert len(inst_end_inds) == 1, "only support one instruction for now"
                labels[last_idx : inst_end_inds[0] + 4 + last_idx] = IGNORE_INDEX
                last_idx = idx + 4
                label_list.append(labels)

    elif model_version == "meta-llama/Llama-3.2-11B-Vision-Instruct":
        image_token_id = LLAMA_IMAGE_INDEX
        # system prompt header "<|start_header_id|>system<|end_header_id|>" has been tokenized to [128006, 9125, 128007]
        # user prompt header "<|start_header_id|>user<|end_header_id|>" has been tokenized to [128006, 882, 128007]
        prompt_header_seqs = torch.tensor(
            [[128006, 9125, 128007], [128006, 882, 128007]]
        )
        assistant_header_seq = torch.tensor([[128006, 78191, 128007]])

        for dialog_tokens in input_ids:
            labels = dialog_tokens.clone()
            last_idx = 0
            eot_indices = (dialog_tokens == LLAMA_EOT_INDEX).nonzero().flatten()

            for idx in eot_indices:
                current_seq = labels[last_idx : idx + 1]
                if len(find_header(prompt_header_seqs, current_seq)) > 0:
                    # found prompt header, indicating that this seq should be masked
                    labels[last_idx : idx + 1] = IGNORE_INDEX
                else:
                    last_idx = idx + 1
            #  Mask all the assistant header prompt <|start_header_id|>assistant<|end_header_id|>, which has been tokenized to [128006, 78191, 128007]
            header_ind = find_header(assistant_header_seq, labels)[0]
            labels[header_ind : header_ind + 3] = IGNORE_INDEX

            label_list.append(labels)

    labels = torch.stack(label_list)
    # Mask the padding token and image token
    pad_or_image_mask = torch.isin(
        labels, torch.tensor([processor.tokenizer.pad_token_id, image_token_id])
    )
    labels[pad_or_image_mask] = IGNORE_INDEX
    return labels
