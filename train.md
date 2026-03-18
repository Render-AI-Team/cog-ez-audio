# Training Data Preparation and Training Guide

## Step 1: Prepare Dataset from Zip

To create a training dataset CSV from a zip file of mp3s (with metadata in the comment property):

````
venv/bin/python src/dataset/prepare_from_zip.py --zip dataset.zip --out_dir extracted_mp3s --csv my_dataset.csv
```

`

- `dataset.zip`: Your zip file containing mp3s (can be nested in folders)
- `extracted_mp3s`: Directory where mp3s will be extracted
- `my_dataset.csv`: Output CSV file for training

## Step 2: Update Config

Edit your config YAML (e.g., src/configs/data_config.yml) to use:

- `data_dir`: extracted_mp3s
- `meta_dir`: my_dataset.csv

## Step 3: Run Training

To start training with Cog:

````
cog train --config-name=src/configs/data_config.yml --epochs=50
```

`

Or directly (if not using Cog):

````
accelerate launch src/train.py --config-name=src/configs/data_config.yml
```

`

## Notes

- The script uses the mp3 "comment" property as the caption for each audio file.
- All nested folders in the zip are handled automatically.
- Make sure mutagen is installed in your Python environment.