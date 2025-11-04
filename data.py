import os
import zipfile
import requests
import collections
import copy
from tqdm import tqdm
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoProcessor


def download_and_extract_data(urls, output_dir='./data'):
    os.makedirs(output_dir, exist_ok=True)
    
    for url in urls:
        filename = url.split('/')[-1]
        filepath = os.path.join(output_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as f, tqdm(
                desc=filename,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for data in response.iter_content(chunk_size=1024):
                    size = f.write(data)
                    pbar.update(size)
        
        extract_dir = os.path.join(output_dir, filename.replace('.zip', ''))
        if not os.path.exists(extract_dir):
            print(f"Extracting {filename}...")
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
    
    return output_dir


def update_filename_column(dataset, path):
    modified_dataset = dataset.map(lambda example: {'filename': path + example['filename']})
    return modified_dataset


def filter_unanswerable(dataset):
    filtered_dataset = dataset.filter(lambda example: example['answer_type'] != 'unanswerable')
    return filtered_dataset


def add_most_common_answer(dataset):
    modified_dataset = copy.deepcopy(dataset)
    max_answer = []
    
    for row in tqdm(modified_dataset, desc='Adding most common answer'):
        answer_list = row['answers']
        answer_counts = collections.Counter(answer_list)
        most_common_answer = answer_counts.most_common(1)[0][0]
        max_answer.append(most_common_answer)
    
    modified_dataset = modified_dataset.add_column('max_answer', max_answer)
    return modified_dataset


class VizWizDataset(Dataset):
    def __init__(self, hf_dataset, processor, image_dir, filter_unanswerable_flag=True):
        self.dataset = hf_dataset
        self.processor = processor
        self.image_dir = image_dir
        
        self.dataset = update_filename_column(self.dataset, image_dir)
        
        if filter_unanswerable_flag:
            self.dataset = filter_unanswerable(self.dataset)
        
        self.dataset = add_most_common_answer(self.dataset)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        image_path = item['filename']
        image = Image.open(image_path).convert('RGB')
        question = item['question']
        answer = item['max_answer']
        
        encoding = self.processor(
            images=image,
            text=question,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        labels = self.processor.tokenizer(
            answer,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        encoding['labels'] = labels.squeeze(0)
        
        return encoding


def create_dataloaders(train_url, val_url, processor_name_or_path, 
                       data_dir='./data', batch_size=8, num_workers=2):
    download_and_extract_data([train_url, val_url], output_dir=data_dir)
    
    train_set = load_dataset("Multimodal-Fatima/VizWiz_train")
    train_hf_dataset = train_set['train']
    
    val_set = load_dataset("Multimodal-Fatima/VizWiz_validation")
    val_hf_dataset = val_set['validation']
    
    processor = AutoProcessor.from_pretrained(processor_name_or_path)
    
    train_image_dir = os.path.join(data_dir, 'train/')
    val_image_dir = os.path.join(data_dir, 'val/')
    
    train_dataset = VizWizDataset(
        train_hf_dataset, 
        processor, 
        train_image_dir, 
        filter_unanswerable_flag=True
    )
    
    val_dataset = VizWizDataset(
        val_hf_dataset, 
        processor, 
        val_image_dir, 
        filter_unanswerable_flag=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def create_inference_dataloader(dataset_name, processor_name_or_path, image_dir, 
                                batch_size=8, num_workers=2, filter_unanswerable_flag=True):
    dataset_mapping = {
        'train': 'Multimodal-Fatima/VizWiz_train',
        'validation': 'Multimodal-Fatima/VizWiz_validation',
        'test': 'Multimodal-Fatima/VizWiz_test'
    }
    
    if dataset_name not in dataset_mapping:
        raise ValueError(f"dataset_name must be one of {list(dataset_mapping.keys())}")
    
    dataset = load_dataset(dataset_mapping[dataset_name])
    hf_dataset = dataset[dataset_name if dataset_name != 'validation' else 'validation']
    
    processor = AutoProcessor.from_pretrained(processor_name_or_path)
    
    vizwiz_dataset = VizWizDataset(
        hf_dataset,
        processor,
        image_dir,
        filter_unanswerable_flag=filter_unanswerable_flag
    )
    
    dataloader = DataLoader(
        vizwiz_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader