# Modification Takes Courage: Seamless Image Stitching via Reference-Driven Inpainting

## Requirements
- Python >= 3.9
- GPU (NVIDIA CUDA compatible) >=24 GB VRAM **If you only have GPUs with 16GB VRAM, we give a low memory plan.**
  
- Create a virtual environment (optional but recommended):

    ```bash
    conda create -n rdistitcher python==3.10
    conda activate rdistitcher
    ```
    
- Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```
## Dataset
 
The UDIS-D dataset, aligned images, and masks can be obtained from  [UDIS](https://github.com/nie-lang/UnsupervisedDeepImageStitching) 
  
The datasets should be organized as follows: 

```
train
├── warp1
│   ├── 000001.jpg
│   ├── ...
├── warp2
│   ├── 000001.jpg
│   ├── ...
├── mask1
│   ├── 000001.jpg
│   ├── ...
├── mask2
│   ├── 000001.jpg
│   ├── ...
```

The test set is in the same format as the training set.

## Train

```bash
    bash train.sh
```

This training configuration requires ~24 GB VRAM with 2 GPUs.

```bash
export MODEL_NAME="stabilityai/stable-diffusion-2-inpainting"
export TRAIN_DIR="path-to-train-set"
export TEST_DIR="path-to-test-set"
export OUTPUT_DIR="RDIStitcherModel"
export CUDA_VISIBLE_DEVICES=0,1

accelerate launch train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$TRAIN_DIR \
  --val_data_dir=$TEST_DIR \
  --output_dir=$OUTPUT_DIR \
  --resolution=512 \
  --train_batch_size=4 \
  --gradient_accumulation_steps=1 \
  --unet_learning_rate=2e-4 \
  --text_encoder_learning_rate=4e-5 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=100 \
  --max_train_steps=50000 \
  --lora_rank=8 \
  --lora_dropout=0.1 \
  --lora_alpha=16 \
  --seed 0 \
  --mixed_precision "no" \
  --identifier "<A>" \
  --tempmodel_steps 10000 \
  --validation_steps 500
```
### Training on a low-memory GPU:

This training configuration requires ~16 GB VRAM with 2 GPUs.

```bash
export MODEL_NAME="stabilityai/stable-diffusion-2-inpainting"
export TRAIN_DIR="path-to-train-set"
export TEST_DIR="path-to-test-set"
export OUTPUT_DIR="RDIStitcherModel"
export CUDA_VISIBLE_DEVICES=0,1

accelerate launch train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$TRAIN_DIR \
  --val_data_dir=$TEST_DIR \
  --output_dir=$OUTPUT_DIR \
  --resolution=512 \
  --train_batch_size=4 \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --use_8bit_adam \
  --enable_xformers_memory_efficient_attention \
  --set_grads_to_none \
  --unet_learning_rate=2e-4 \
  --text_encoder_learning_rate=4e-5 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=100 \
  --max_train_steps=50000 \
  --lora_rank=8 \
  --lora_dropout=0.1 \
  --lora_alpha=16 \
  --seed 0 \
  --mixed_precision "no" \
  --identifier "<A>" \
  --tempmodel_steps 10000 \
  --validation_steps 500
```

## Test

Our pre-trained LoRA weights are very small at only 12MB, so you can use it directly in the `loraweight` document. Due to hardware limitations, we cannot give the best "LoRA" setting, but we think the presented pre-trained LoRA weight is sufficient to demonstrate the effectiveness of our work.

```bash
    bash test.sh
```

This testing configuration requires ~6 GB VRAM.

```bash
python inference.py \
    --device "cuda" \
    --pretrained_model_name_or_path "stabilityai/stable-diffusion-2-inpainting" \
    --loramodel_path "loraweight" \
    --data_root "path-to-test-set" \
    --num_seed 5
```

## MLLM-based Metrics

For using qwen:
 ```bash
    pip install openai
 ```
For using glm:
 ```bash
    pip install zhipuai
 ```

### SIQS

```bash
python mllmmetrics.py \
    --metric_type "qwen-siqs" or "glm-siqs" \
    --image_path "path-to-stitched-images" \
    --api_key "your-api_key" \
    --base_url "your-base-url" \
```

### MICQS

```bash
python mllmmetrics.py \
    --metric_type "qwen-micqs" or "glm-micqs" \
    --image_path "path-to-stitched-images" \
    --image_path2 "path2-to-stitched-images" \
    --api_key "your-api_key" \
    --base_url "your-base-url" \
```
