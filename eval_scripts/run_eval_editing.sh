#!/bin/bash
CUDA_VISIBLE_DEVICES=5,6 \
    PYTHONPATH=/home/w4756677/garment/garment_foundation_model/sewformer/SewFormer:/home/w4756677/garment/garment_foundation_model/sewformer/SewFactory/packages \
    torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    eval_scripts/eval_llava.py \
    experiment.run_name=garmentcodedata.llava-7b.regression.r8.kv.lr6e-4.pos.sinusoidal_0.multimodal.eval.editing \
    trainer.steps_per_epoch=750 \
    trainer.epoches=20 \
    data_wrapper.data_split.load_from_split_file=/home/w4756677/garment/garment_foundation_model/sewformer/SewFormer/assets/data_configs/garmentcodedata_datasplit.json \
    'data_wrapper.dataset.sampling_rate=[0,0,0,0,1]' data_wrapper.dataset.editing_flip_prob=0  \
    model.num_freq=0 pre_trained=/miele/george/garment/sewformer/outputs/2024-10-23/15-56-57/ckpt_16 \
    data_wrapper/dataset=qva_garment_token_dataset_garmentcodedata \
    data_wrapper/dataset/garment_tokenizer/standardize=garmentcodedata_regression_stats \
    hydra.run.dir=/miele/george/garment/sewformer/outputs/eval_editing \
    gen_only=True eval_val=True eval_train=False model_max_length=2100 lora_args.lora_r=0  \
    --config-name train_garment_llava_regression_w_pos --config-path ../configs