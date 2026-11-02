"""
Verification and extraction script for foal_c2_backbone_results.zip.
"""
import hashlib
import json
import os
from pathlib import Path
import zipfile

def verify_and_extract_c2_zip(zip_path: Path, extract_dir: Path) -> dict:
    if not zip_path.exists():
        raise FileNotFoundError(f"Missing ZIP archive: {zip_path}")

    # Compute SHA256
    sha256_hash = hashlib.sha256()
    size_bytes = zip_path.stat().st_size
    with open(zip_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256_hash.update(chunk)
    zip_sha256 = sha256_hash.hexdigest()

    # Inspect ZIP entries
    with zipfile.ZipFile(zip_path, "r") as zf:
        # Test ZIP integrity (CRC32 checks)
        bad_file = zf.testzip()
        crc_valid = (bad_file is None)

        namelist = zf.namelist()
        total_uncompressed_bytes = sum(info.file_size for info in zf.infolist())

        # Check for duplicates or unsafe paths
        duplicate_entries = len(namelist) - len(set(namelist))
        suspicious_paths = [name for name in namelist if name.startswith(("/", "\\", "..")) or ":" in name]

        # Safe extraction
        extract_dir.mkdir(parents=True, exist_ok=True)
        extracted_files = 0
        extracted_dirs = set()
        for member in zf.infolist():
            target_path = extract_dir / member.filename
            # Prevent path traversal
            if not target_path.resolve().is_relative_to(extract_dir.resolve()):
                raise ValueError(f"Malicious path traversal detected: {member.filename}")
            zf.extract(member, extract_dir)
            if member.is_dir():
                extracted_dirs.add(member.filename)
            else:
                extracted_files += 1

    summary = {
        "zip_filename": zip_path.name,
        "size_bytes": size_bytes,
        "size_mb": round(size_bytes / (1024 ** 2), 2),
        "sha256": zip_sha256,
        "entries_count": len(namelist),
        "total_uncompressed_bytes": total_uncompressed_bytes,
        "total_uncompressed_mb": round(total_uncompressed_bytes / (1024 ** 2), 2),
        "crc_valid": crc_valid,
        "crc_first_bad_file": bad_file,
        "duplicate_entries": duplicate_entries,
        "suspicious_paths": len(suspicious_paths),
        "extracted_file_count": extracted_files,
        "extracted_dir_count": len(extracted_dirs),
        "namelist": sorted(namelist),
    }

    summary_path = Path("results/c2_full_gpu_verified_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary

if __name__ == "__main__":
    zip_p = Path("foal_c2_backbone_results.zip")
    ext_p = Path("results/c2_full_gpu_verified")
    res = verify_and_extract_c2_zip(zip_p, ext_p)
    print("=" * 60)
    print(f"ZIP Archive: {res['zip_filename']}")
    print(f"Size       : {res['size_mb']} MB ({res['size_bytes']} bytes)")
    print(f"SHA256     : {res['sha256']}")
    print(f"Entries    : {res['entries_count']} files/dirs")
    print(f"CRC Valid  : {res['crc_valid']}")
    print(f"Uncompressed: {res['total_uncompressed_mb']} MB")
    print(f"Extracted to: {ext_p}")
    print("=" * 60)
