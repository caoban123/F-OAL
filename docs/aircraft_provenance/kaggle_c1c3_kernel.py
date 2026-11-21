import csv, glob, hashlib, json, os, re, shutil, subprocess, sys, time
from collections import Counter

W = '/kaggle/working'; T = '/kaggle/tmp'; R = f'{T}/foal'
os.makedirs(T, exist_ok=True)
TEAM_COMMIT = 'b0f3223'
AUTHOR_COMMIT = '73e4ac1b4c965030973a2b77f05c57c2f44f49f5'
SEED = 42


def sh(c, **k):
    print('$', c, flush=True)
    return subprocess.run(c, shell=True, check=True, **k)


sh('pip install -q torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126')
sh('pip install -q timm==1.0.30 datasets wandb pyyaml pytest')
sh(f'git clone -q https://github.com/caoban123/F-OAL.git {R}')
os.chdir(R)
sh(f'git checkout -q {TEAM_COMMIT}')
sh('git rev-parse HEAD')
sh('git status --short')
import torch, yaml
print('GPU', torch.cuda.get_device_name(0), torch.__version__, flush=True)
os.environ.update(WANDB_MODE='offline', WANDB_DIR=W, WANDB_BASE_URL='https://api.forge.coreweave.com', PYTHONPATH=R)

base = yaml.safe_load(open('configs/aircraft/aircraft_a0_full.yaml'))
IN1K = 'vit_base_patch16_224.augreg_in1k'
IN21K = 'vit_base_patch16_224.augreg2_in21k_ft_in1k'
# name, group, backbone, interpolation, normalize, projection init, tags
RUNS = [
    ('c1_p0_gaussian_std1', 'c1_projection_initialization_aircraft', IN1K, 'bicubic', True, 'gaussian_std1', 'c1,projection-init,gaussian-std1'),
    ('c1_p1_official_uniform', 'c1_projection_initialization_aircraft', IN1K, 'bicubic', True, 'official_uniform', 'c1,projection-init,official-uniform'),
    ('c1_p2_gaussian_matched_variance', 'c1_projection_initialization_aircraft', IN1K, 'bicubic', True, 'gaussian_matched_variance', 'c1,projection-init,gaussian-matched-variance'),
    ('c2_b1_augreg2_in21k_ft_in1k', 'c2_backbone_alignment_aircraft', IN21K, 'bicubic', True, 'official_uniform', 'c2,backbone,augreg2-in21k-ft-in1k'),
    ('c3_b_bilinear_normalized', 'c3_preprocessing_alignment_aircraft', IN21K, 'bilinear', True, 'official_uniform', 'c3,preprocessing,bilinear,normalized'),
    ('c3_c_bicubic_raw', 'c3_preprocessing_alignment_aircraft', IN21K, 'bicubic', False, 'official_uniform', 'c3,preprocessing,bicubic,raw'),
]
rows = []
for name, group, bb, interp, norm, init, tags in RUNS:
    c = json.loads(json.dumps(base))
    c['dataset'].update(interpolation=interp, normalize=norm,
                        mean=[0.5, 0.5, 0.5] if norm else None, std=[0.5, 0.5, 0.5] if norm else None)
    c['model'].update(backbone=bb, projection_init=init)
    c['projection'].update(initialization=init)
    c['logging'].update(group=group)
    cfg = f'{T}/{name}.yaml'
    yaml.safe_dump(c, open(cfg, 'w'))
    rn = f'aircraft_{name}_seed{SEED}'; out = f'{W}/runs/{rn}'
    t = time.time()
    sh(f'python scripts/train.py --config {cfg} --seed {SEED} --class-order-seed {SEED} --projection-seed {SEED} --stream-seed {SEED} '
       f'--experiment-type {"controlled-c1" if name.startswith("c1") else "additional-analysis"} --feature-cache-dir {T}/cache --run-name {rn} --output-dir {out} '
       f'--wandb --wandb-offline --wandb-group {group} --wandb-tags aircraft,controlled-experiment,{tags},seed{SEED}',
       stdout=subprocess.DEVNULL)
    shutil.copy(cfg, f'{out}/config_used.yaml')
    m = json.load(open(f'{out}/metrics.json')); m.pop('task_matrix', None)
    r = {'name': name, 'seed': SEED, 'sec': round(time.time() - t), **{k: v for k, v in m.items() if isinstance(v, (int, float))}}
    rows.append(r); print(json.dumps(r), flush=True)
    for f in glob.glob(f'{out}/**/*.pt', recursive=True):
        os.remove(f)
with open(f'{W}/summary_c1c3.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, sorted({k for x in rows for k in x})); w.writeheader(); w.writerows(rows)
shutil.rmtree(f'{T}/cache', ignore_errors=True)

# ---- Author code on Aircraft, unmodified, seed 0 (same way the team ran it on CIFAR-100) ----
O = f'{T}/official-F-OAL'; SHIM = f'{T}/shim'
sh(f'git clone -q https://github.com/liuyuchen-cz/F-OAL.git {O}')
sh(f'git -C {O} checkout -q {AUTHOR_COMMIT}')
sh(f'git -C {O} status --short')
os.makedirs(SHIM, exist_ok=True)
open(f'{SHIM}/loss.py', 'w').write(
    "# Import shim: author repo imports loss.py for the DVC agent, which F-OAL does not use.\n"
    "def agmax_loss(*a, **k):\n    raise NotImplementedError\n"
    "def cross_entropy_loss(*a, **k):\n    raise NotImplementedError\n")
sh('pip install -q timm==0.9.7 kornia scikit-learn scipy')
os.chdir(O)
env = dict(os.environ, PYTHONPATH=f'{SHIM}:{O}')
t = time.time()
p = subprocess.run(f'python general_main.py --data aircraft --cl_type nc --agent FOAL --num_runs 1 --seed 0',
                   shell=True, env=env, capture_output=True, text=True)
official_sec = time.time() - t
os.makedirs(f'{W}/official_foal_aircraft_seed0', exist_ok=True)
open(f'{W}/official_foal_aircraft_seed0/official_stdout.log', 'w').write(p.stdout + '\n--- stderr ---\n' + p.stderr)
print('official exit', p.returncode, 'sec', round(official_sec), flush=True)
print(p.stdout[-3000:], flush=True); print(p.stderr[-3000:], flush=True)
for f in glob.glob(f'{O}/**/*.csv', recursive=True):
    shutil.copy(f, f'{W}/official_foal_aircraft_seed0/')
# accuracy rows printed after each task: "[0.966 0. ...]"
mat = []
for line in p.stdout.splitlines():
    s = line.strip()
    if s.startswith('[') and s.endswith(']') and len(s.split()) >= 10 and not re.search(r'[a-zA-Z]', s):
        mat.append([float(x) for x in s.strip('[]').split()])
mat = mat[:10]
print('official matrix rows', len(mat), flush=True)
if len(mat) == 10:
    import numpy as np, wandb
    A = np.array(mat)
    stage = [float(A[i, :i + 1].mean()) for i in range(10)]
    avg = float(np.mean(stage)); last = stage[-1]
    f_past = float(np.mean([A[j:9, j].max() - A[9, j] for j in range(9)]))
    f_code = float(np.mean([max(0.0, A[j:, j].max() - A[9, j]) for j in range(10)]))
    res = {'average_incremental_accuracy': avg, 'last_accuracy': last, 'forgetting_past_tasks': f_past,
           'forgetting_official_code': f_code, 'stage_accuracies': stage, 'task_matrix': mat,
           'total_duration_seconds': official_sec}
    json.dump(res, open(f'{W}/official_foal_aircraft_seed0/metrics.json', 'w'), indent=2)
    json.dump({'method': 'official_foal', 'run_name': 'official_foal_aircraft_seed0',
               'official_repo_url': 'https://github.com/liuyuchen-cz/F-OAL', 'official_commit_sha': AUTHOR_COMMIT,
               'command': 'python general_main.py --data aircraft --cl_type nc --agent FOAL --num_runs 1 --seed 0',
               'shim': 'loss.py (DVC agent import only)', 'timm': '0.9.7', 'gpu': torch.cuda.get_device_name(0)},
              open(f'{W}/official_foal_aircraft_seed0/metadata.json', 'w'), indent=2)
    run = wandb.init(project='hcmus-foal-reproduction', entity='caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm',
                     group='official_reproduction_aircraft', name='official_foal_aircraft_seed0',
                     tags=['aircraft', 'official', 'reproduction', 'seed_0'], dir=W,
                     config={'method': 'official_foal', 'dataset': 'fgvc_aircraft', 'seed': 0,
                             'official_commit_sha': AUTHOR_COMMIT, 'timm': '0.9.7'})
    for i, a in enumerate(stage):
        run.log({'task': i + 1, 'stage_accuracy_pct': a * 100})
    run.summary.update({'summary/average_incremental_accuracy_pct': avg * 100, 'summary/last_accuracy_pct': last * 100,
                        'summary/forgetting_past_tasks_pct': f_past * 100,
                        'summary/forgetting_official_code_pct': f_code * 100,
                        'summary/total_duration_seconds': official_sec})
    run.finish()
    print('OFFICIAL', json.dumps({k: v for k, v in res.items() if not isinstance(v, list)}), flush=True)

# ---- HF _test split vs torchvision split="test" (label + pixels), CPU only ----
try:
    from datasets import load_dataset
    from torchvision.datasets import FGVCAircraft
    from PIL import Image
    def md5(im):
        return hashlib.md5(im.convert('RGB').tobytes()).hexdigest()
    hf = load_dataset('Multimodal-Fatima/FGVC_Aircraft_test', split='test').select_columns(['image', 'label'])
    n = hf.features['label'].names
    b = Counter((n[x['label']], md5(x['image'])) for x in hf)
    for split in ('test', 'val'):
        tv = FGVCAircraft(f'{T}/tv', split=split, annotation_level='variant', download=True)
        a = Counter((tv.classes[l], md5(Image.open(f))) for f, l in zip(tv._image_files, tv._labels))
        print('CROSSCHECK_TEST hf_test', len(hf), f'tv_{split}', len(tv), 'same(label+pixels)', sum((a & b).values()), flush=True)
except Exception as e:
    print('CROSSCHECK_TEST failed', repr(e), flush=True)
print('DONE', flush=True)
