import os
from pathlib import Path

def main():
    wandb_dir = Path("results/c3_full_gpu_verified/wandb")
    for r in sorted(wandb_dir.iterdir()):
        if r.is_dir() and r.name.startswith("offline-run-"):
            debug_log = r / "logs" / "debug.log"
            content = ""
            if debug_log.exists():
                with open(debug_log, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            
            # Find any mentions of c3_ or pilot or smoke
            run_name = "Unknown"
            for token in ["c3_b_bilinear_normalized", "c3_c_bicubic_raw", "c3_d_bilinear_raw", "pilot", "smoke"]:
                if token in content:
                    run_name = token
                    break
            
            # Also check wandb file size
            wandb_files = list(r.glob("*.wandb"))
            wsize = wandb_files[0].stat().st_size if wandb_files else 0
            
            print(f"{r.name}: detected='{run_name}', wandb_bytes={wsize}")

if __name__ == "__main__":
    main()
