#!/usr/bin/env python3
"""
Dataset Downloader for Real Task Execution
Downloads and prepares real datasets in MEDIUM, LARGE, and HUGE sizes
"""

import os
import requests
from pathlib import Path
import subprocess
import json
from PIL import Image
import numpy as np
import pandas as pd

class DatasetDownloader:
    def __init__(self, base_dir='data/real_datasets'):
        self.base_dir = Path(base_dir)
        self.images_dir = self.base_dir / 'images'
        self.videos_dir = self.base_dir / 'videos'
        self.csv_dir = self.base_dir / 'csv'
        self.text_dir = self.base_dir / 'text'
        self.logs_dir = self.base_dir / 'logs'
        
        # Create directories
        for dir in [self.images_dir, self.videos_dir, self.csv_dir, 
                    self.text_dir, self.logs_dir]:
            dir.mkdir(parents=True, exist_ok=True)
    
    def download_sample_images(self):
        """Download sample images from internet"""
        print("📥 Downloading sample images...")
        
        # Unsplash sample images (free to use)
        image_urls = [
            "https://source.unsplash.com/4000x3000/?nature,1",
            "https://source.unsplash.com/4000x3000/?city,2",
            "https://source.unsplash.com/4000x3000/?people,3",
            "https://source.unsplash.com/4000x3000/?food,4",
            "https://source.unsplash.com/4000x3000/?technology,5",
        ]
        
        for i, url in enumerate(image_urls):
            try:
                response = requests.get(url, timeout=30)
                if response.status_code == 200:
                    filepath = self.images_dir / f'sample_{i+1}.jpg'
                    with open(filepath, 'wb') as f:
                        f.write(response.content)
                    print(f"   ✅ Downloaded {filepath.name}")
            except Exception as e:
                print(f"   ⚠️  Failed to download image {i+1}: {e}")
    
    def generate_synthetic_images(self):
        """Generate synthetic images of various sizes"""
        print("\n🎨 Generating synthetic images...")
        
        sizes = {
            'MEDIUM': [(2000, 1500), 50],   # 50 images
            'LARGE': [(3000, 2250), 100],   # 100 images
            'HUGE': [(4000, 3000), 200]     # 200 images
        }
        
        for size_cat, (dimensions, count) in sizes.items():
            size_dir = self.images_dir / size_cat.lower()
            size_dir.mkdir(exist_ok=True)
            
            for i in range(count):
                # Create random image
                img_array = np.random.randint(0, 256, 
                                               (*dimensions, 3), 
                                               dtype=np.uint8)
                img = Image.fromarray(img_array)
                
                filepath = size_dir / f'image_{i:04d}.jpg'
                img.save(filepath, quality=95)
                
                if (i + 1) % 20 == 0:
                    print(f"   Generated {i+1}/{count} {size_cat} images")
        
        print(f"   ✅ Generated images for all size categories")
    
    def generate_csv_datasets(self):
        """Generate realistic CSV datasets"""
        print("\n📊 Generating CSV datasets...")
        
        def generate_sales_data(rows):
            return pd.DataFrame({
                'date': pd.date_range('2023-01-01', periods=rows, freq='H'),
                'product_id': np.random.randint(1000, 9999, rows),
                'category': np.random.choice(['Electronics', 'Clothing', 
                                               'Food', 'Books', 'Home'], rows),
                'quantity': np.random.randint(1, 100, rows),
                'price': np.random.uniform(10, 1000, rows),
                'customer_id': np.random.randint(10000, 99999, rows),
                'region': np.random.choice(['North', 'South', 'East', 'West'], rows),
                'revenue': np.random.uniform(50, 5000, rows)
            })
        
        def generate_customer_data(rows):
            return pd.DataFrame({
                'customer_id': range(rows),
                'name': [f'Customer_{i}' for i in range(rows)],
                'age': np.random.randint(18, 80, rows),
                'income': np.random.randint(20000, 200000, rows),
                'credit_score': np.random.randint(300, 850, rows),
                'purchases_last_year': np.random.randint(0, 500, rows),
                'total_spent': np.random.uniform(0, 50000, rows),
                'signup_date': pd.date_range('2020-01-01', periods=rows, freq='H')
            })
        
        def generate_stock_data(rows):
            return pd.DataFrame({
                'timestamp': pd.date_range('2023-01-01', periods=rows, freq='min'),
                'ticker': np.random.choice(['AAPL', 'GOOGL', 'MSFT', 
                                            'AMZN', 'TSLA'], rows),
                'open': np.random.uniform(100, 500, rows),
                'high': np.random.uniform(100, 500, rows),
                'low': np.random.uniform(100, 500, rows),
                'close': np.random.uniform(100, 500, rows),
                'volume': np.random.randint(1000000, 100000000, rows)
            })
        
        datasets = {
            'MEDIUM': {
                'sales_50mb.csv': (generate_sales_data, 200000),
                'customers_30mb.csv': (generate_customer_data, 150000),
                'stocks_40mb.csv': (generate_stock_data, 180000)
            },
            'LARGE': {
                'sales_150mb.csv': (generate_sales_data, 600000),
                'customers_100mb.csv': (generate_customer_data, 500000),
                'stocks_120mb.csv': (generate_stock_data, 550000)
            },
            'HUGE': {
                'sales_400mb.csv': (generate_sales_data, 1600000),
                'customers_300mb.csv': (generate_customer_data, 1500000),
                'stocks_350mb.csv': (generate_stock_data, 1550000)
            }
        }
        
        for size_cat, files in datasets.items():
            size_dir = self.csv_dir / size_cat.lower()
            size_dir.mkdir(exist_ok=True)
            
            for filename, (generator, rows) in files.items():
                print(f"   Generating {filename} ({size_cat})...")
                df = generator(rows)
                filepath = size_dir / filename
                df.to_csv(filepath, index=False)
                
                size_mb = filepath.stat().st_size / (1024 * 1024)
                print(f"   ✅ Created {filename} ({size_mb:.1f} MB)")
    
    def generate_text_datasets(self):
        """Generate text datasets for NLP tasks"""
        print("\n📝 Generating text datasets...")
        
        def generate_text(words):
            """Generate random text"""
            sample_words = ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 
                           'lazy', 'dog', 'and', 'runs', 'through', 'forest',
                           'data', 'processing', 'machine', 'learning', 'artificial',
                           'intelligence', 'computer', 'science', 'algorithm']
            return ' '.join(np.random.choice(sample_words, words))
        
        text_sizes = {
            'MEDIUM': 5_000_000,    # ~5 million words (~25MB)
            'LARGE': 15_000_000,    # ~15 million words (~75MB)
            'HUGE': 40_000_000      # ~40 million words (~200MB)
        }
        
        for size_cat, word_count in text_sizes.items():
            size_dir = self.text_dir / size_cat.lower()
            size_dir.mkdir(exist_ok=True)
            
            print(f"   Generating {size_cat} text ({word_count:,} words)...")
            
            filepath = size_dir / f'text_corpus_{size_cat.lower()}.txt'
            
            # Generate in chunks to avoid memory issues
            chunk_size = 100000
            with open(filepath, 'w') as f:
                for i in range(0, word_count, chunk_size):
                    chunk = generate_text(min(chunk_size, word_count - i))
                    f.write(chunk + '\n')
            
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"   ✅ Created {filepath.name} ({size_mb:.1f} MB)")
    
    def generate_log_files(self):
        """Generate realistic log files"""
        print("\n📋 Generating log files...")
        
        log_levels = ['INFO', 'WARNING', 'ERROR', 'DEBUG']
        components = ['API', 'Database', 'Cache', 'Worker', 'Scheduler']
        messages = [
            'Request processed successfully',
            'Database query executed',
            'Cache hit',
            'Task completed',
            'Connection established',
            'Timeout occurred',
            'Retry attempt',
            'Invalid input detected'
        ]
        
        def generate_log_line():
            timestamp = pd.Timestamp.now().isoformat()
            level = np.random.choice(log_levels)
            component = np.random.choice(components)
            message = np.random.choice(messages)
            return f"{timestamp} [{level}] {component}: {message}\n"
        
        log_sizes = {
            'MEDIUM': 500_000,      # ~500k lines (~50MB)
            'LARGE': 1_500_000,     # ~1.5M lines (~150MB)
            'HUGE': 4_000_000       # ~4M lines (~400MB)
        }
        
        for size_cat, line_count in log_sizes.items():
            size_dir = self.logs_dir / size_cat.lower()
            size_dir.mkdir(exist_ok=True)
            
            print(f"   Generating {size_cat} logs ({line_count:,} lines)...")
            
            filepath = size_dir / f'application_{size_cat.lower()}.log'
            
            with open(filepath, 'w') as f:
                for i in range(line_count):
                    f.write(generate_log_line())
                    if (i + 1) % 100000 == 0:
                        print(f"      {i+1:,}/{line_count:,} lines")
            
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"   ✅ Created {filepath.name} ({size_mb:.1f} MB)")
    
    def generate_summary(self):
        """Generate summary of all datasets"""
        print("\n📊 Dataset Summary:")
        print("=" * 60)
        
        total_size = 0
        file_count = 0
        
        for subdir in [self.images_dir, self.videos_dir, self.csv_dir,
                       self.text_dir, self.logs_dir]:
            if subdir.exists():
                size = sum(f.stat().st_size for f in subdir.rglob('*') if f.is_file())
                count = sum(1 for f in subdir.rglob('*') if f.is_file())
                size_mb = size / (1024 * 1024)
                
                total_size += size
                file_count += count
                
                print(f"{subdir.name:15s}: {count:4d} files, {size_mb:8.1f} MB")
        
        total_mb = total_size / (1024 * 1024)
        total_gb = total_mb / 1024
        
        print("=" * 60)
        print(f"{'TOTAL':15s}: {file_count:4d} files, {total_mb:8.1f} MB ({total_gb:.2f} GB)")
        
        # Save summary
        summary = {
            'total_files': file_count,
            'total_size_mb': total_mb,
            'total_size_gb': total_gb,
            'directories': {
                'images': sum(1 for f in self.images_dir.rglob('*') if f.is_file()),
                'videos': sum(1 for f in self.videos_dir.rglob('*') if f.is_file()),
                'csv': sum(1 for f in self.csv_dir.rglob('*') if f.is_file()),
                'text': sum(1 for f in self.text_dir.rglob('*') if f.is_file()),
                'logs': sum(1 for f in self.logs_dir.rglob('*') if f.is_file())
            }
        }
        
        with open(self.base_dir / 'dataset_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✅ Summary saved to {self.base_dir}/dataset_summary.json")
    
    def download_all(self):
        """Download and generate all datasets"""
        print("🚀 Starting dataset preparation...")
        print("=" * 60)
        
        # Download sample images
        self.download_sample_images()
        
        # Generate datasets
        self.generate_synthetic_images()
        self.generate_csv_datasets()
        self.generate_text_datasets()
        self.generate_log_files()
        
        # Generate summary
        self.generate_summary()
        
        print("\n" + "=" * 60)
        print("✅ Dataset preparation complete!")
        print("=" * 60)


if __name__ == '__main__':
    downloader = DatasetDownloader()
    downloader.download_all()
