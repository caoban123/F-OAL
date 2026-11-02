"""
Update all documentation files with verified W&B run links, metadata, and standardized evidence.
"""

from pathlib import Path

# 1. Update README.md
readme_path = Path("README.md")
readme_text = readme_path.read_text(encoding="utf-8")

old_c1_line = '| **F-OAL Controlled (`C1` Projection)** | Ours (`src/foal`) | CIFAR-100 | `42` | **83.28%** (P1) / **83.37%** (P2) / **79.33%** (P0) | **74.84%** (P1) / **74.99%** (P2) / **70.95%** (P0) | **8.07%** / **8.97%** (P1) & **7.93%** / **8.81%** (P2) | `269f05c` | [`results/c1_analysis/`](results/c1_analysis/) | Colab Offline | `VERIFIED FULL GPU` (`C1 PASS`) |'

new_c1_lines = """| **C1: P0 Gaussian std=1** | Ours (`src/foal`) | CIFAR-100 | `42` (`n=1, std=N/A`) | **79.33%** | **70.95%** | **8.25%** / **9.17%** | `269f05c` | [`results/c1_analysis/`](results/c1_analysis/) | [Run `qw78us1k`](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/qw78us1k) | `VERIFIED FULL GPU` (`C1 PASS`) |
| **C1: P1 Official Uniform** | Ours (`src/foal`) | CIFAR-100 | `42` (`n=1, std=N/A`) | **83.28%** | **74.84%** | **8.07%** / **8.97%** | `269f05c` | [`results/c1_analysis/`](results/c1_analysis/) | [Run `6whrot4z`](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/6whrot4z) | `VERIFIED FULL GPU` (`C1 PASS`) |
| **C1: P2 Matched Gaussian** | Ours (`src/foal`) | CIFAR-100 | `42` (`n=1, std=N/A`) | **83.37%** | **74.99%** | **7.93%** / **8.81%** | `269f05c` | [`results/c1_analysis/`](results/c1_analysis/) | [Run `hsgsvb6p`](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/hsgsvb6p) | `VERIFIED FULL GPU` (`C1 PASS`) |"""

if old_c1_line in readme_text:
    readme_text = readme_text.replace(old_c1_line, new_c1_lines)
    readme_path.write_text(readme_text, encoding="utf-8", newline="\n")
    print("Updated README.md")
else:
    print("Warning: old_c1_line not found in README.md")

# 2. Update README_IMPLEMENTATION.md
impl_path = Path("README_IMPLEMENTATION.md")
impl_text = impl_path.read_text(encoding="utf-8")

old_impl_c1 = '| **F-OAL Controlled (`C1`–`C4`)** | Ours (`src/foal`) | CIFAR-100 | `42` / `0` | `NOT RUN` | `NOT RUN` | `NOT RUN` | `PENDING` | [`docs/original_vs_ours_protocol.md`](docs/original_vs_ours_protocol.md) | `NOT YET SYNCED` | `IN PROGRESS` (`A20`) |'

new_impl_c1 = """| **C1: P0 Gaussian std=1** | Ours (`src/foal`) | CIFAR-100 | `42` (`n=1, std=N/A`) | **79.33%** | **70.95%** | **8.25%** / **9.17%** | `269f05c` | [`results/c1_analysis/`](results/c1_analysis/) | [Run `qw78us1k`](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/qw78us1k) | `VERIFIED FULL GPU` (`C1 PASS`) |
| **C1: P1 Official Uniform** | Ours (`src/foal`) | CIFAR-100 | `42` (`n=1, std=N/A`) | **83.28%** | **74.84%** | **8.07%** / **8.97%** | `269f05c` | [`results/c1_analysis/`](results/c1_analysis/) | [Run `6whrot4z`](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/6whrot4z) | `VERIFIED FULL GPU` (`C1 PASS`) |
| **C1: P2 Matched Gaussian** | Ours (`src/foal`) | CIFAR-100 | `42` (`n=1, std=N/A`) | **83.37%** | **74.99%** | **7.93%** / **8.81%** | `269f05c` | [`results/c1_analysis/`](results/c1_analysis/) | [Run `hsgsvb6p`](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/hsgsvb6p) | `VERIFIED FULL GPU` (`C1 PASS`) |"""

impl_text = impl_text.replace("33 unit tests", "47 unit tests")
if old_impl_c1 in impl_text:
    impl_text = impl_text.replace(old_impl_c1, new_impl_c1)

impl_links_append = """* **C1 P0 Gaussian std=1 (`qw78us1k`, Seed 42, Commit `269f05c`):** [https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/qw78us1k](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/qw78us1k)
* **C1 P1 Official Uniform (`6whrot4z`, Seed 42, Commit `269f05c`):** [https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/6whrot4z](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/6whrot4z)
* **C1 P2 Matched Gaussian (`hsgsvb6p`, Seed 42, Commit `269f05c`):** [https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/hsgsvb6p](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/hsgsvb6p)
"""
if "hsgsvb6p" not in impl_text:
    impl_text += "\n" + impl_links_append

impl_path.write_text(impl_text, encoding="utf-8", newline="\n")
print("Updated README_IMPLEMENTATION.md")

# 3. Update docs/experiments.md
exp_path = Path("docs/experiments.md")
exp_text = exp_path.read_text(encoding="utf-8")

old_wandb_target = '| `c1_cifar100_gaussian_std1_seed42` | `c1_cifar100_official_uniform_seed42` | `c1_cifar100_gaussian_matched_variance_seed42` | — | — | — |'
new_wandb_target = '| [Run `qw78us1k`](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/qw78us1k) | [Run `6whrot4z`](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/6whrot4z) | [Run `hsgsvb6p`](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/hsgsvb6p) | — | — | — |'

if old_wandb_target in exp_text:
    exp_text = exp_text.replace(old_wandb_target, new_wandb_target)
    exp_path.write_text(exp_text, encoding="utf-8", newline="\n")
    print("Updated docs/experiments.md")
else:
    print("Warning: old_wandb_target not found in docs/experiments.md")

# 4. Update docs/original_vs_ours_protocol.md
proto_path = Path("docs/original_vs_ours_protocol.md")
proto_text = proto_path.read_text(encoding="utf-8")

old_proto_p0 = '- **P0 (`gaussian_std1`):** $\\bar{A} = \\mathbf{79.33\\%}$, $A_{10} = \\mathbf{70.95\\%}$, $F_{10}^{\\text{past}} = \\mathbf{9.17\\%}$ (tái hiện hoàn hảo bản chạy lịch sử $79.33\\%$).'
new_proto_p0 = '- **P0 (`gaussian_std1`):** $\\bar{A} = \\mathbf{79.33\\%}$, $A_{10} = \\mathbf{70.95\\%}$, $F_{10}^{\\text{past}} = \\mathbf{9.17\\%}$, $F_{10}^{\\text{code}} = \\mathbf{8.25\\%}$ ([W&B Run `qw78us1k`](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/qw78us1k)).'

old_proto_p1 = '- **P1 (`official_uniform`):** $\\bar{A} = \\mathbf{83.28\\%}$, $A_{10} = \\mathbf{74.84\\%}$, $F_{10}^{\\text{past}} = \\mathbf{8.97\\%}$ ($\\Delta_{P1 - P0} = \\mathbf{+3.95\\text{ pp}}$ trên $\\bar{A}$).'
new_proto_p1 = '- **P1 (`official_uniform`):** $\\bar{A} = \\mathbf{83.28\\%}$, $A_{10} = \\mathbf{74.84\\%}$, $F_{10}^{\\text{past}} = \\mathbf{8.97\\%}$, $F_{10}^{\\text{code}} = \\mathbf{8.07\\%}$ ($\\Delta_{P1 - P0} = \\mathbf{+3.95\\text{ pp}}$ trên $\\bar{A}$, [W&B Run `6whrot4z`](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/6whrot4z)).'

old_proto_p2 = '- **P2 (`gaussian_matched_variance`):** $\\bar{A} = \\mathbf{83.37\\%}$, $A_{10} = \\mathbf{74.99\\%}$, $F_{10}^{\\text{past}} = \\mathbf{8.81\\%}$ ($\\Delta_{P2 - P0} = \\mathbf{+4.04\\text{ pp}}$ trên $\\bar{A}$).'
new_proto_p2 = '- **P2 (`gaussian_matched_variance`):** $\\bar{A} = \\mathbf{83.37\\%}$, $A_{10} = \\mathbf{74.99\\%}$, $F_{10}^{\\text{past}} = \\mathbf{8.81\\%}$, $F_{10}^{\\text{code}} = \\mathbf{7.93\\%}$ ($\\Delta_{P2 - P0} = \\mathbf{+4.04\\text{ pp}}$ trên $\\bar{A}$, [W&B Run `hsgsvb6p`](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/hsgsvb6p)).'

proto_text = proto_text.replace(old_proto_p0, new_proto_p0)
proto_text = proto_text.replace(old_proto_p1, new_proto_p1)
proto_text = proto_text.replace(old_proto_p2, new_proto_p2)
proto_path.write_text(proto_text, encoding="utf-8", newline="\n")
print("Updated docs/original_vs_ours_protocol.md")

# 5. Update docs/FOAL_AUDIT_AND_FIX_PLAN.md
plan_path = Path("docs/FOAL_AUDIT_AND_FIX_PLAN.md")
plan_text = plan_path.read_text(encoding="utf-8")

old_plan_p0 = '- **P0 (`gaussian_std1`):** $\\bar{A} = 79.33\\%$, $A_{10} = 70.95\\%$, $F_{10}^{\\text{past}} = 9.17\\%$, $F_{10}^{\\text{code}} = 8.25\\%$.'
new_plan_p0 = '- **P0 (`gaussian_std1`):** $\\bar{A} = 79.33\\%$, $A_{10} = 70.95\\%$, $F_{10}^{\\text{past}} = 9.17\\%$, $F_{10}^{\\text{code}} = 8.25\\%$ ([W&B Run `qw78us1k`](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/qw78us1k)).'

old_plan_p1 = '- **P1 (`official_uniform`):** $\\bar{A} = 83.28\\%$, $A_{10} = 74.84\\%$, $F_{10}^{\\text{past}} = 8.97\\%$, $F_{10}^{\\text{code}} = 8.07\\%$ ($\\Delta_{P1-P0} = +3.95\\text{ pp}$).'
new_plan_p1 = '- **P1 (`official_uniform`):** $\\bar{A} = 83.28\\%$, $A_{10} = 74.84\\%$, $F_{10}^{\\text{past}} = 8.97\\%$, $F_{10}^{\\text{code}} = 8.07\\%$ ($\\Delta_{P1-P0} = +3.95\\text{ pp}$, [W&B Run `6whrot4z`](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/6whrot4z)).'

old_plan_p2 = '- **P2 (`gaussian_matched_variance`):** $\\bar{A} = 83.37\\%$, $A_{10} = 74.99\\%$, $F_{10}^{\\text{past}} = 8.81\\%$, $F_{10}^{\\text{code}} = 7.93\\%$ ($\\Delta_{P2-P0} = +4.04\\text{ pp}$).'
new_plan_p2 = '- **P2 (`gaussian_matched_variance`):** $\\bar{A} = 83.37\\%$, $A_{10} = 74.99\\%$, $F_{10}^{\\text{past}} = 8.81\\%$, $F_{10}^{\\text{code}} = 7.93\\%$ ($\\Delta_{P2-P0} = +4.04\\text{ pp}$, [W&B Run `hsgsvb6p`](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/hsgsvb6p)).'

plan_text = plan_text.replace(old_plan_p0, new_plan_p0)
plan_text = plan_text.replace(old_plan_p1, new_plan_p1)
plan_text = plan_text.replace(old_plan_p2, new_plan_p2)
plan_path.write_text(plan_text, encoding="utf-8", newline="\n")
print("Updated docs/FOAL_AUDIT_AND_FIX_PLAN.md")
