# Vision Language Garment Models
This is the codebase used for the experiments in the work
"Towards Vision Language Garment Models" presented at the Multimodal Foundation Models workshop at CVPR'25.

## Setup
Start by creating a virtual environment and installing the relevant packages. If you use `conda` please install the environment from the `environment.yml` file in the root dir.

### Dataset
You can download the dataset from the same place as our previous work [AIpparel](https://github.com/georgeNakayama/AIpparel-Code).

### Checkpoint
We provide a checkpoint of a trained version of the model on [huggingface](https://huggingface.co/ackermannj/vision-language-garment-model).

### Config
Under the folder `configs` please configure all the entries that are currently `<>`.

## Running the Model
We provide the same entry for training and evaluation. All runs can be started from `training_scripts/train_vlg_llama3.py`.
Here is an example how to start the training on a single GPU node with 8 gpus.
```
WORKDIR=$(pwd)
RUN_NAME="Test"

PYTHONPATH=$WORKDIR:$WORKDIR/src torchrun --nnodes=1 --nproc_per_node=8 training_scripts/train_vlg_llama3.py \
    --config-name train_v2 \
    run_name="$RUN_NAME" \
    project="vlg-train" \
    dataset=vlg_dataset
```

To resume or evaluate an existing checkpoint add the flag `+resume=<>` pointing to your checkpoint.

## Citation

If you are using our model or dataset in your project, consider citing our paper.

```
@article{ackermann2025vlg,
    title={Towards Vision-Language-Garment Models For Web Knowledge Garment Understanding and Generation},
    author={Ackermann, Jan and Nakayama, Kiyohiro and Yang, Guandao and Wu, Tong and Wetzstein, Gordon},
    journal={CVPRW},
    year={2025}
}
```