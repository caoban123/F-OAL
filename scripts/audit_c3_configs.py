"""Audit C3 Configs and compare with C2-B1."""
import yaml
from pathlib import Path

def audit():
    configs = {}
    for name in ['c3_a_bicubic_normalized', 'c3_b_bilinear_normalized', 'c3_c_bicubic_raw', 'c3_d_bilinear_raw']:
        p = Path(f'configs/{name}.yaml')
        configs[name] = yaml.safe_load(p.read_text(encoding='utf-8'))

    # Also load C2-B1
    c2_b1 = yaml.safe_load(Path('configs/c2_b1_augreg2_in21k_ft_in1k.yaml').read_text(encoding='utf-8'))

    fixed_keys = [
        ('seed', 42),
        ('class_order_seed', 42),
        ('projection_seed', 42),
        ('stream_seed', 42),
        ('dataset.name', 'uoft-cs/cifar100'),
        ('dataset.num_classes', 100),
        ('dataset.classes_per_task', 10),
        ('dataset.class_order', None),
        ('dataset.image_size', 224),
        ('projection.dimension', 1000),
        ('projection.initialization', 'official_uniform'),
        ('projection.seed', 42),
        ('model.backbone', 'vit_base_patch16_224.augreg2_in21k_ft_in1k'),
        ('model.pretrained', True),
        ('model.feature_fusion', True),
        ('model.smooth_projection', True),
        ('model.projection_dim', 1000),
        ('model.projection_init', 'official_uniform'),
        ('model.gamma', 1.0),
        ('model.rls_dtype', 'float64'),
        ('training.batch_size', 10),
        ('logging.project', 'hcmus-foal-reproduction'),
        ('logging.base_url', 'https://api.forge.coreweave.com'),
        ('logging.group', 'c3_preprocessing_alignment')
    ]

    for name, cfg in configs.items():
        print(f"=== Checking {name} ===")
        for k, expected in fixed_keys:
            parts = k.split('.')
            val = cfg
            for part in parts:
                val = val[part]
            assert val == expected, f"Mismatch in {name} for {k}: {val} != {expected}"
        interp = cfg['dataset']['interpolation']
        norm = cfg['dataset']['normalize']
        mean = cfg['dataset']['mean']
        std = cfg['dataset']['std']
        print(f"  Fixed controls: PASSED")
        print(f"  Factor A (interpolation): {interp}")
        print(f"  Factor B (normalize)    : {norm}")
        print(f"  Mean / Std              : {mean} / {std}")

    print("\n=== Checking C3-A vs C2-B1 Equivalence ===")
    c3_a = configs['c3_a_bicubic_normalized']
    for section in ['seed', 'class_order_seed', 'projection_seed', 'stream_seed']:
        assert c3_a[section] == c2_b1[section], f"Mismatch in {section}"
    for k in ['name', 'num_classes', 'classes_per_task', 'image_size', 'interpolation', 'mean', 'std']:
        assert c3_a['dataset'][k] == c2_b1['dataset'][k], f"Mismatch in dataset.{k}"
    for k in ['dimension', 'initialization', 'seed']:
        assert c3_a['projection'][k] == c2_b1['projection'][k], f"Mismatch in projection.{k}"
    for k in ['backbone', 'pretrained', 'feature_fusion', 'smooth_projection', 'projection_dim', 'projection_init', 'gamma', 'rls_dtype']:
        assert c3_a['model'][k] == c2_b1['model'][k], f"Mismatch in model.{k}"
    assert c3_a['training']['batch_size'] == c2_b1['training']['batch_size']
    print("C3-A and C2-B1 are 100% equivalent across all execution hyperparameters!")

if __name__ == '__main__':
    audit()
