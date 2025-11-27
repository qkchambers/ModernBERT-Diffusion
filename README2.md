# Install

# Run
* Create a yaml for each configuration
    - BERT Normal
    - KANBERT
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
* Skipping weight decay because there notebook does not use it


# Kan
* Replaced the Linear layer in attention and the mlp layer with kan, also removed activation function
    - Left prediction layer the same, so maybe change that
* init_weights
    - skipping over kan in this function since kan does not have a weights/bias parameter to use


# Diffusion
* not doing the 1% of batches with random sequence length since we are just doing classification and not generation
* Removed a potential optimization in the masked model forward function that is being used for diffusion
* Eval is done the same way as the masked language model.

## TODO
* look into masking specifics for modernbert
* add eps as a config option?
* remove unescesary diffusion config options
* split code for github
* Using accuracy in stsb crashes out :(


# BERT
n_params=1.3300e+07
started:6:16pm
ended:7:28pm

# BERT full mask
started:4:31pm


# Diffusion
n_params=1.3300e+07
started:9:55pm 1/2 batch size

# KAN
* Would be 1.3329e+08 with same configuration
    - So using that decoder block added 100M parameters
    
* Final: 1.3872e+07

# Results
