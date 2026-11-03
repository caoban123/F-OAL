import hashlib
import zipfile
import os
import json
from pathlib import Path

zip_path = Path('foal_table4_ablation_results.zip')
assert zip_path.exists(), 'Zip file not found!'

size = zip_path.stat().st_size
hasher = hashlib.sha256()
with open(zip_path, 'rb') as f:
    while chunk := f.read(65536):
        hasher.update(chunk)
sha256 = hasher.hexdigest()

manifest = {
    'archive_path': str(zip_path.resolve()),
    'size_bytes': size,
    'sha256': sha256,
    'entries': []
}

with zipfile.ZipFile(zip_path, 'r') as zf:
    test_res = zf.testzip()
    manifest['crc32_status'] = 'ALL_OK' if test_res is None else f'FAILED_AT_{test_res}'
    infolist = zf.infolist()
    manifest['total_entries'] = len(infolist)
    manifest['compressed_bytes'] = sum(i.compress_size for i in infolist)
    manifest['uncompressed_bytes'] = sum(i.file_size for i in infolist)

    for info in infolist:
        is_unsafe = info.filename.startswith('/') or '..' in info.filename or os.path.isabs(info.filename)
        manifest['entries'].append({
            'filename': info.filename,
            'file_size': info.file_size,
            'compress_size': info.compress_size,
            'crc32': hex(info.CRC),
            'is_unsafe': is_unsafe
        })

output_dir = Path('results/table4_analysis')
output_dir.mkdir(parents=True, exist_ok=True)
with open(output_dir / 'table4_archive_manifest.json', 'w', encoding='utf-8') as f:
    json.dump(manifest, f, indent=2)

print("Archive inspection complete!")
print("Size:", size)
print("SHA256:", sha256)
print("CRC Status:", manifest['crc32_status'])
print("Total entries:", manifest['total_entries'])
for e in manifest['entries']:
    print(f" - {e['filename']} ({e['file_size']} bytes)")
