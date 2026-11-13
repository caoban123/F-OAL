"""Write DTD cache manifest if not already present."""
from pathlib import Path
import json
from src.foal.dtd_datasets import make_dtd_class_order

cache_dir = Path('results/dtd_feature_cache')
cache_dir.mkdir(parents=True, exist_ok=True)
order = make_dtd_class_order(num_classes_to_use=40, total_classes=47, seed=42)
selected_classes = sorted(list(set(order)))

manifest = {
    'num_classes': 40,
    'selected_classes': selected_classes,
    'class_order': order,
    'train_samples_per_class': 40,
    'test_samples_per_class': 40,
    'seed': 42,
    'primary_backbone': 'vit_base_patch16_224.augreg2_in21k_ft_in1k',
    'control_backbone': 'vit_base_patch16_224.augreg_in1k',
    'cache_files': [f.name for f in cache_dir.glob('*.pt')],
}

manifest_file = cache_dir / 'cache_manifest.json'
manifest_file.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print(f"Manifest written to {manifest_file.resolve()}")
print(f"Class order (len={len(order)}): {order}")
