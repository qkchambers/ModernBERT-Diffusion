# Install

# Run
* Create a yaml for each configuration
    - BERT Normal
    - Diffusion

# Dataloader
python src/convert_dataset.py --dataset Skylion007/openwebtext --out_root ./openwebtext-data --splits train val
* might be wrong
https://github.com/mosaicml/examples/tree/main/examples/benchmarks/bert#prepare-your-data

# Pretrain
composer main.py yamls/modernbert/modernbert-base-pretrain-diffusion.yaml


# Convert from .pt to hugging face model


# Finetune
* Use finetune_modernbert_on_glue.ipynb


# Benchmarks
* Each of the glue tasks
    - Maybe use the same as ModernBERT
* Track training/validation loss
* Track total time to run / or time per epoch or something
* number of parameters
* hyperparameters



# Diffusion
* not doing the 1% of batches with random sequence length since we are just doing classification and not generation
* Eval is done the same way as the masked language model.

# BERT
n_params=1.3300e+07
started:6:16pm
ended:7:28pm

