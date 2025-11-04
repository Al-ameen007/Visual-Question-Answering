# BLIP-VQA Training on VizWiz Dataset

Train BLIP Visual Question Answering model on the VizWiz dataset.

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Training (with defaults)

```bash
python train.py
```

### Custom Training

```bash
python train.py \
    --num_epochs 5 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --save_every_epoch \
    --plot_training
```

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--train_url` | str | `https://vizwiz.cs.colorado.edu/VizWiz_final/images/train.zip` | Training data URL |
| `--val_url` | str | `https://vizwiz.cs.colorado.edu/VizWiz_final/images/val.zip` | Validation data URL |
| `--model_name` | str | `Salesforce/blip-vqa-base` | Pretrained model |
| `--data_dir` | str | `./data` | Data directory |
| `--save_dir` | str | `./checkpoints` | Checkpoint directory |
| `--num_epochs` | int | `3` | Number of epochs |
| `--batch_size` | int | `16` | Batch size |
| `--learning_rate` | float | `5e-5` | Learning rate |
| `--weight_decay` | float | `0.01` | Weight decay |
| `--num_workers` | int | `4` | Data loader workers |
| `--eval_freq` | int | `100` | Evaluation frequency |
| `--eval_iter` | int | `50` | Eval batch count |
| `--save_every_epoch` | flag | False | Save after each epoch |
| `--plot_training` | flag | False | Generate training plot |

## Examples

### Train for 10 epochs with larger batch size

```bash
python train.py --num_epochs 10 --batch_size 32
```

### Train with custom learning rate and save checkpoints

```bash
python train.py \
    --learning_rate 1e-4 \
    --num_epochs 5 \
    --save_every_epoch \
    --plot_training
```

### Train on custom data directory

```bash
python train.py \
    --data_dir /path/to/data \
    --save_dir /path/to/checkpoints
```

### Full training with all options

```bash
python train.py \
    --model_name Salesforce/blip-vqa-base \
    --num_epochs 10 \
    --batch_size 32 \
    --learning_rate 5e-5 \
    --weight_decay 0.01 \
    --num_workers 8 \
    --eval_freq 50 \
    --eval_iter 100 \
    --save_every_epoch \
    --plot_training
```

## Output

After training, you'll find:

- `./checkpoints/final_model/` - Final trained model
- `./checkpoints/best_model/` - Best model (lowest val loss)
- `./checkpoints/checkpoint-epoch-*.pt` - Per-epoch checkpoints (if `--save_every_epoch`)
- `training_curve.png` - Training plot (if `--plot_training`)

## Results

The model was trained for 5 epochs on the VizWiz dataset. Performance was evaluated using the accuracy metric specified by the VizWiz dataset creators:

```
acc = min(#Humans_that_answered_that_answer / 3, 1)
```

Implementation details: Accuracy was calculated using the Universal Sentence Encoder from TensorFlow Hub. Embeddings of predicted and candidate answers were generated, and cosine similarity was computed. A predicted answer was considered correct if its maximum similarity score with candidate answers was ≥ 0.8.

### Training Progress

| Epoch | Training Loss |
|-------|---------------|
| 1     | 3.267         |
| 2     | 2.509         |
| 3     | 2.118         |
| 4     | 1.901         |
| 5     | 1.778         |

### Accuracy on VizWiz Validation Set

| Model | Overall Accuracy | Yes/No Accuracy | Number Accuracy | Other Accuracy |
|-------|------------------|-----------------|-----------------|----------------|
| **Base Model** | 33.69% | 74.36% | 31.0% | 15.28% |
| **Trained Model** | **50.92%** | **82.56%** | **48.77%** | **43.06%** |
| LXR955 (SOTA) | 55.4% | 74.0% | 39.0% | 24.76% |

**Key Findings:**

- **+17.23%** overall accuracy improvement over base model
- **+8.2%** improvement on yes/no questions
- **+17.77%** improvement on number questions  
- **+27.78%** improvement on "other" category questions
- Achieved **92.1%** of state-of-the-art (LXR955) performance
- Outperformed SOTA on yes/no questions (**+8.56%**)

*Note: LXR955 is the current best model on the VizWiz dataset according to Papers with Code.*

## Loading Trained Model

```python
from transformers import BlipForQuestionAnswering, AutoProcessor

# Load from HuggingFace Hub (pre-trained model)
model = BlipForQuestionAnswering.from_pretrained('MohammadAlameenArtan/BLIP_Model_VizWiz')
processor = AutoProcessor.from_pretrained('MohammadAlameenArtan/BLIP_Model_VizWiz')

# Or load from local checkpoint
model = BlipForQuestionAnswering.from_pretrained('./checkpoints/best_model')
processor = AutoProcessor.from_pretrained('./checkpoints/best_model')
```

## Inference

### Simple Inference

Run inference on a single image using the pre-trained model from HuggingFace:

```bash
# Using URL (uses HuggingFace model by default)
python inference.py \
    --image "https://example.com/image.jpg" \
    --question "What is in the picture?"

# Using local file
python inference.py \
    --image "./path/to/image.jpg" \
    --question "What color is the car?"

# Using your own trained model
python inference.py \
    --image "image.jpg" \
    --question "How many people?" \
    --model_path ./checkpoints/final_model
```

### Compare Base vs Trained Model

Compare predictions between base and trained models (uses HuggingFace model by default):

```bash
# Basic comparison (trained model from HuggingFace)
python inference_compare.py \
    --image "https://example.com/image.jpg" \
    --question "What vehicles are in the picture?"

# With correct answer and visualization
python inference_compare.py \
    --image "https://example.com/car.jpg" \
    --question "What color is the car?" \
    --correct_answer "red" \
    --visualize

# Save visualization
python inference_compare.py \
    --image "image.jpg" \
    --question "What is this?" \
    --correct_answer "cat" \
    --save_viz comparison.png

# Use your own trained model instead
python inference_compare.py \
    --image "image.jpg" \
    --question "What is this?" \
    --trained_model ./checkpoints/best_model
```

## Programmatic Inference

```python
from inference import load_model, load_image, predict_answer

# Load model from HuggingFace Hub
model, processor, device = load_model('MohammadAlameenArtan/BLIP_Model_VizWiz')

# Or load from local checkpoint
# model, processor, device = load_model('./checkpoints/best_model')

# Load image (URL or file path)
image = load_image('https://example.com/image.jpg')

# Get prediction
answer = predict_answer(image, "What is in the image?", model, processor, device)
print(f"Answer: {answer}")
```

## Project Structure

```
.
├── train.py              # Training script (CLI)
├── inference.py          # Simple inference (CLI)
├── inference_compare.py  # Compare base vs trained (CLI)
├── engine.py             # Training engine
├── data.py               # Data loading
├── requirements.txt      # Dependencies
├── example_usage.py      # Programmatic usage example
└── README.md             # This file
```
