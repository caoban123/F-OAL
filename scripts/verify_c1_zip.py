import hashlib
import json
from pathlib import Path
import sys
import zipfile

# Ensure stdout handles UTF-8 safely
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

zip_path = Path("foal_c1_projection_results.zip")
extract_dir = Path("results/c1_full_gpu_verified")
extract_dir.mkdir(parents=True, exist_ok=True)

print(f"Inspecting: {zip_path.name}")
file_size = zip_path.stat().st_size
print(f"File size: {file_size} bytes ({file_size / (1024**2):.2f} MB)")

print("Computing SHA256...")
h = hashlib.sha256()
with open(zip_path, "rb") as f:
    while chunk := f.read(1024 * 1024):
        h.update(chunk)
sha256_hex = h.hexdigest()
print(f"SHA256: {sha256_hex}")

print("Testing ZIP CRC integrity...")
with zipfile.ZipFile(zip_path, "r") as zf:
    bad_file = zf.testzip()
    if bad_file:
        print(f"ERROR: Corrupt file detected in ZIP: {bad_file}")
    else:
        print("CRC Test: PASSED (No corrupt files)")

    infolist = zf.infolist()
    namelist = zf.namelist()
    print(f"Total entries in ZIP: {len(infolist)}")
    total_uncompressed = sum(info.file_size for info in infolist)
    print(f"Total uncompressed size: {total_uncompressed} bytes ({total_uncompressed / (1024**2):.2f} MB)")

    duplicates = len(namelist) - len(set(namelist))
    print(f"Duplicate entry names: {duplicates}")

    # Check suspicious paths
    suspicious = [name for name in namelist if name.startswith(("/", "\\", "..")) or ":" in name]
    print(f"Suspicious paths: {len(suspicious)}")

    print("\n--- ZIP NAMELIST SAMPLE (first 10 and last 10) ---")
    for name in namelist[:10]:
        print(" ", name)
    if len(namelist) > 20:
        print("  ...")
    for name in namelist[-10:]:
        print(" ", name)

    print(f"\nExtracting safely into results/c1_full_gpu_verified...")
    zf.extractall(extract_dir)
    print("Extraction complete!")

extracted_files = list(extract_dir.rglob("*"))
file_count = sum(1 for f in extracted_files if f.is_file())
dir_count = sum(1 for f in extracted_files if f.is_dir())
print(f"\nExtracted verification: {file_count} files, {dir_count} directories.")

summary = {
    "zip_filename": zip_path.name,
    "size_bytes": file_size,
    "size_mb": round(file_size / (1024**2), 2),
    "sha256": sha256_hex,
    "entries_count": len(infolist),
    "total_uncompressed_bytes": total_uncompressed,
    "total_uncompressed_mb": round(total_uncompressed / (1024**2), 2),
    "crc_valid": bad_file is None,
    "duplicate_entries": duplicates,
    "suspicious_paths": len(suspicious),
    "extracted_file_count": file_count,
    "extracted_dir_count": dir_count,
    "namelist": namelist,
}

Path("results/c1_full_gpu_verified_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print("Saved summary to results/c1_full_gpu_verified_summary.json")
