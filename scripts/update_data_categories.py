import json
from pathlib import Path

def update_dataset_categories():
    data_path = Path('data/task_profiles_clean_final.json')
    if not data_path.exists():
        print("❌ Dataset not found.")
        return

    with open(data_path, 'r') as f:
        data = json.load(f)

    # Mapping based on task_type
    mapping = {
        'matrix_multiplication': 'compute',
        'monte_carlo_simulation': 'compute',
        'statistical_analysis': 'compute',
        'csv_correlation_analysis': 'analysis',
        'log_parsing': 'analysis',
        'log_pattern_matching': 'analysis',
        'csv_aggregation': 'analysis',
        'csv_groupby': 'analysis',
        'csv_merge_operations': 'io_heavy',
        'data_deduplication': 'io_heavy',
        'log_aggregation': 'io_heavy',
        'text_search_replace': 'io_heavy',
        'text_tokenization': 'io_heavy',
        'text_word_count': 'io_heavy',
        'image_compression': 'media',
        'image_filter_application': 'media',
        'image_resize': 'media',
        'thumbnail_generation': 'media'
    }

    print(f"🔄 Updating {len(data)} records in {data_path}...")
    for rec in data:
        t_type = rec.get('task_type')
        if t_type in mapping:
            rec['task_category'] = mapping[t_type]

    with open(data_path, 'w') as f:
        json.dump(data, f, indent=2)

    print("✅ Dataset task_category updated.")

if __name__ == '__main__':
    update_dataset_categories()
