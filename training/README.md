# SelfCite Preference Optimization with SimPO

## Setup

Install the dependencies with pip:
```
pip install torch==2.4.1
pip install flashinfer-python -i https://flashinfer.ai/whl/cu121/torch2.4
pip install -r requirements.txt
```

or try to create a conda env from `training-env.yaml`: (notice that sometimes the installation fails due to the support of `torch` and some version conflicts)
```
conda env create -f training-env.yaml -n selfcite
conda activate selfcite
```

## Training

Generate the preference pairs using BoN with the following steps. If you want to skip this step, a pre-generated data can be downloaded from [here](https://www.dropbox.com/scl/fi/dpv6zyjsaw4vgdhjsn7d5/selfcite-train-2k.json?rlkey=dn5j2l3klnyk0e9w0cccgn5hv&st=ribd3ch8&dl=0), which contains 2k preference pairs of BoN sampled results from LongCite-8B.

```bash
HF_DATASETS_CACHE=[path to cache] ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file accelerate_configs/deepspeed_zero3.yaml scripts/run_simpo.py \
    training_configs/longcite-8b-simpo.yaml \
    --dataset_mixer="selfcite-train-2k.json:1.0" \
    --run_name=debug \
    --learning_rate=3e-7 \
    --output_dir=tmp_debug
```

## Generate training data with BoN

```
pip install nltk
```

### Step 1: Sampling candidates

If you have multiple nodes, e.g. 100 nodes, you can add the following arguments to split the examples into 100 shards and run them independently.

```bash
HF_DATASETS_CACHE=[path to cache] python truncate_irrelevant_and_sampling.py \
    --shard_id $TASK_ID \ # 0-99
    --total_shards 100 \
    --model_path THUDM/LongCite-llama3.1-8b \
    --save_dir $YOUR_SAMPLING_OUTPUT_DIR \
    --num_gpus $NUM_GPUS
```

### Step 2: Reranking candidates

```bash
HF_DATASETS_CACHE=[path to cache] python truncate_irrelevant_and_reranking.py \
    --shard_id $TASK_ID \ # 0-99
    --total_shards 100 \
    --model_path THUDM/LongCite-llama3.1-8b \
    --sampling_path $YOUR_SAMPLING_OUTPUT_DIR \
    --save_path $YOUR_RERANKING_OUTPUT_DIR \
    --num_gpus $NUM_GPUS
```

## Step 3: Generate preference pairs

```bash
python make_preference_data.py "$YOUR_RERANKING_OUTPUT_DIR/shard_*_out_of_*/tmp/*.jsonl" [output_file]
```