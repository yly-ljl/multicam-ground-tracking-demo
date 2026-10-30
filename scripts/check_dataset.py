"""Check full WILDTRACK extraction, archive CRCs, filenames and prior samples."""
import csv
from datetime import datetime
import filecmp
import json
from pathlib import Path
import time
import zipfile
import zlib

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/setup_check'
ARCHIVE = ROOT / 'data/downloads/Wildtrack_dataset_full.zip'
DATA = ROOT / 'data/Wildtrack_dataset'


def main():
    started = time.perf_counter()
    download = json.loads((OUT / 'mirror_download.json').read_text(encoding='utf-8'))
    assert download['status'] == 'verified', 'Full mirror download has not passed SHA256 verification'
    assert ARCHIVE.stat().st_size == 6807496358
    with zipfile.ZipFile(ARCHIVE) as archive:
        all_files = [item for item in archive.infolist() if not item.is_dir()]
    # Only read central-directory metadata with zipfile. Its header offsets are not used.
    skipped = [item for item in all_files
               if '__MACOSX' in Path(item.filename).parts or Path(item.filename).name == '.DS_Store']
    skipped_names = {item.filename for item in skipped}
    included = [item for item in all_files if item.filename not in skipped_names]
    actual_paths = {path.relative_to(ROOT / 'data').as_posix()
                    for path in DATA.rglob('*') if path.is_file()}
    assert not any(path.is_file() for path in (ROOT / 'data/__MACOSX').rglob('*'))
    expected_paths = {item.filename for item in included}
    assert actual_paths == expected_paths, {
        'missing': sorted(expected_paths - actual_paths), 'extra': sorted(actual_paths - expected_paths)}
    actual_bytes = 0
    with (OUT / 'dataset_file_checks.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['path', 'bytes', 'crc32', 'expected_crc32', 'match'])
        writer.writeheader()
        for index, item in enumerate(included, 1):
            path = ROOT / 'data' / item.filename
            size = path.stat().st_size
            assert size == item.file_size, (item.filename, size, item.file_size)
            crc = 0
            with path.open('rb') as source:
                for block in iter(lambda: source.read(1024 * 1024), b''):
                    crc = zlib.crc32(block, crc)
            crc &= 0xffffffff
            writer.writerow({'path': item.filename, 'bytes': size, 'crc32': f'{crc:08x}',
                             'expected_crc32': f'{item.CRC:08x}', 'match': crc == item.CRC})
            assert crc == item.CRC, f'CRC mismatch: {item.filename}'
            actual_bytes += size
            if index % 500 == 0:
                print(f'CRC checked {index}/{len(included)} files', flush=True)

    image_ids = [f'{number:08d}' for number in range(0, 2001, 5)]
    annotation_ids = [f'{number:08d}' for number in range(0, 2000, 5)]
    annotations = sorted(path.stem for path in (DATA / 'annotations_positions').glob('*.json'))
    assert annotations == annotation_ids
    cameras = {}
    for camera in range(1, 8):
        name = f'C{camera}'
        ids = sorted(path.stem for path in (DATA / 'Image_subsets' / name).glob('*.png'))
        assert ids == image_ids, name
        cameras[name] = {'images': len(ids), 'first': ids[0], 'last': ids[-1], 'step': 5,
                         'aligned_annotations': len(set(ids) & set(annotations)),
                         'images_without_gt': sorted(set(ids) - set(annotations))}
    calibration_files = sorted(path.relative_to(DATA).as_posix()
                               for path in (DATA / 'calibrations').rglob('*') if path.is_file())
    assert len(calibration_files) == 21, calibration_files
    sample_matches = []
    subset = ROOT / 'data/setup_subset/Wildtrack_dataset'
    for sample in sorted(path for path in subset.rglob('*') if path.is_file()):
        relative = sample.relative_to(subset)
        same = filecmp.cmp(sample, DATA / relative, shallow=False)
        assert same, relative
        sample_matches.append({'path': relative.as_posix(), 'bytes': sample.stat().st_size,
                               'byte_for_byte_equal': same})
    assert len(sample_matches) == 4
    report = {
        'status': 'passed', 'recorded_at': datetime.now().astimezone().isoformat(),
        'seconds': time.perf_counter() - started,
        'archive': str(ARCHIVE.relative_to(ROOT)), 'archive_bytes': ARCHIVE.stat().st_size,
        'data_root': str(DATA.relative_to(ROOT)),
        'archive_declared_files': len(all_files),
        'archive_declared_uncompressed_bytes': sum(item.file_size for item in all_files),
        'excluded_macos_files': len(skipped), 'excluded_macos_bytes': sum(item.file_size for item in skipped),
        'excluded_rule': '__MACOSX/** and all .DS_Store files; no dataset image/annotation/calibration excluded.',
        'extracted_files': len(included), 'actual_extracted_bytes': actual_bytes,
        'crc_checked_files': len(included), 'crc_failures': 0,
        'cameras': cameras, 'total_images': sum(value['images'] for value in cameras.values()),
        'annotations': {'files': len(annotations), 'first': annotations[0], 'last': annotations[-1], 'step': 5},
        'calibration_files': calibration_files, 'prior_official_samples': sample_matches,
        'content_access': 'Byte streams read only for decompression, CRC and equality of the four previously reviewed samples. No held-out image decoding or annotation parsing.',
        'source_verification': 'Whole archive matches mirror-published SHA256. Official source equality was checked for size, final 400000 bytes and four initial samples, not the whole official archive.',
    }
    (OUT / 'dataset_check.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
