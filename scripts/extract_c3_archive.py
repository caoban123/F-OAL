import os
import sys
import zipfile
import hashlib
import json
from pathlib import Path

def main():
    zip_path = Path("foal_c3_preprocessing_results.zip")
    extract_dir = Path("results/c3_full_gpu_verified")
    extract_dir.mkdir(parents=True, exist_ok=True)

    with open(zip_path, "rb") as f:
        zip_bytes = f.read()
        zip_sha256 = hashlib.sha256(zip_bytes).hexdigest()

    manifest = {
        "archive_path": str(zip_path.resolve()),
        "archive_size_bytes": len(zip_bytes),
        "archive_sha256": zip_sha256,
        "entries_count": 0,
        "total_uncompressed_bytes": 0,
        "crc_valid": True,
        "suspicious_paths": 0,
        "extracted_files": []
    }

    with zipfile.ZipFile(zip_path, "r") as z:
        infolist = z.infolist()
        manifest["entries_count"] = len(infolist)
        
        # Check CRC
        bad_file = z.testzip()
        if bad_file:
            manifest["crc_valid"] = False
            manifest["first_bad_file"] = bad_file
            print(f"CRC check failed at: {bad_file}")
        else:
            print("CRC check passed for all files!")

        for info in infolist:
            manifest["total_uncompressed_bytes"] += info.file_size
            
            # Check path traversal
            norm_p = os.path.normpath(info.filename)
            if norm_p.startswith("..") or os.path.isabs(norm_p):
                manifest["suspicious_paths"] += 1
                print(f"Suspicious path: {info.filename}")
                continue
                
            target_path = extract_dir / info.filename
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            if not info.is_dir():
                with z.open(info) as src, open(target_path, "wb") as dst:
                    data = src.read()
                    dst.write(data)
                    file_sha256 = hashlib.sha256(data).hexdigest()
                    manifest["extracted_files"].append({
                        "name": info.filename,
                        "size": info.file_size,
                        "sha256": file_sha256
                    })

    print(f"Extracted {len(manifest['extracted_files'])} files to {extract_dir}")

    analysis_dir = Path("results/c3_analysis")
    analysis_dir.mkdir(parents=True, exist_ok=True)
    with open(analysis_dir / "c3_archive_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("Saved results/c3_analysis/c3_archive_manifest.json successfully!")

if __name__ == "__main__":
    main()
