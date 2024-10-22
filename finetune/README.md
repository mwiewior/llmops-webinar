
## Llama-3.1-8B-Instruct
```bash
cd finetune
MODEL="Meta-Llama-3.1-8B-Instruct"
MODEL_HF="meta-llama/$MODEL"
MODEL_PATH=/tmp/$MODEL
mkdir -p $MODEL_PATH
tune download $MODEL_HF --output-dir $MODEL_PATH --ignore-patterns "*.pth" --hf-token xxx
tune cp llama3_1/8B_qlora_single_device ./custom_config.yaml
```


## Qwen
```bash
MODEL=Qwen2.5-3B-Instruct
MODEL_HF="Qwen/"$MODEL
MODEL_PATH=/tmp/$MODEL
mkdir -p $MODEL_PATH
tune download $MODEL_HF --output-dir $MODEL_PATH --ignore-patterns "*.pth"
```