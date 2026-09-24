"""Extract the RFID data-collection archives in data_files/raw_data.

Each .zip is extracted into its own directory under the output root, named
after the archive stem, e.g.

    data_files/extracted/Rev121-Layout02_2026-09-03/...

The archives are not consistent about their internal layout (some have a
top-level folder, some start straight at test_<n>_<date>/), so keeping one
directory per archive avoids collisions between them.

Usage:
    python unzip_raw_data.py                      # extract all, skip done ones
    python unzip_raw_data.py --force              # re-extract everything
    python unzip_raw_data.py --src DIR --out DIR
    python unzip_raw_data.py --list               # show contents, extract nothing
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

# /home/ups/backend/preprocessing/unzip_raw_data.py -> /home/ups
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SRC = PROJECT_ROOT / "data_files" / "raw_data"
DEFAULT_OUT = PROJECT_ROOT / "data_files" / "extracted"


def is_within(base: Path, target: Path) -> bool:
    """True if target stays inside base (blocks ../ and absolute members)."""
    try:
        target.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def safe_extract(zf: zipfile.ZipFile, dest: Path) -> int:
    """Extract every member into dest, refusing anything that escapes it."""
    dest.mkdir(parents=True, exist_ok=True)
    count = 0
    for member in zf.infolist():
        name = member.filename
        if name.startswith("/") or ".." in Path(name).parts:
            raise ValueError(f"unsafe path in archive: {name!r}")
        if not is_within(dest, dest / name):
            raise ValueError(f"member escapes destination: {name!r}")
        zf.extract(member, dest)
        if not member.is_dir():
            count += 1
    return count


def extract_one(zip_path: Path, out_root: Path, force: bool = False) -> Path:
    """Extract one archive into out_root/<zip stem>/ and return that path."""
    dest = out_root / zip_path.stem
    marker = dest / ".extracted"

    if marker.exists() and not force:
        print(f"[skip] {zip_path.name} -> {dest} (already extracted)")
        return dest

    print(f"[open] {zip_path.name} ({zip_path.stat().st_size / 1e6:.1f} MB)")
    with zipfile.ZipFile(zip_path) as zf:
        bad = zf.testzip()
        if bad is not None:
            raise zipfile.BadZipFile(f"corrupt entry {bad!r} in {zip_path.name}")
        n_files = safe_extract(zf, dest)

    marker.write_text(f"{zip_path.name}\n{n_files} files\n")
    print(f"[done] {zip_path.name} -> {dest} ({n_files} files)")
    return dest


def list_one(zip_path: Path, limit: int = 20) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    print(f"\n=== {zip_path.name} ({len(names)} members)")
    for name in names[:limit]:
        print(f"    {name}")
    if len(names) > limit:
        print(f"    ... {len(names) - limit} more")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC,
                        help=f"directory holding the .zip files (default: {DEFAULT_SRC})")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help=f"where to extract (default: {DEFAULT_OUT})")
    parser.add_argument("--force", action="store_true",
                        help="re-extract archives that were already extracted")
    parser.add_argument("--list", action="store_true", dest="list_only",
                        help="print archive contents instead of extracting")
    args = parser.parse_args(argv)

    src: Path = args.src.expanduser().resolve()
    if not src.is_dir():
        print(f"error: source directory not found: {src}", file=sys.stderr)
        return 1

    zips = sorted(src.glob("*.zip"))
    if not zips:
        print(f"error: no .zip files in {src}", file=sys.stderr)
        return 1

    if args.list_only:
        for zip_path in zips:
            list_one(zip_path)
        return 0

    out_root: Path = args.out.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    failures: list[tuple[Path, Exception]] = []
    for zip_path in zips:
        try:
            extract_one(zip_path, out_root, force=args.force)
        except (zipfile.BadZipFile, ValueError, OSError) as exc:
            print(f"[fail] {zip_path.name}: {exc}", file=sys.stderr)
            failures.append((zip_path, exc))

    print(f"\n{len(zips) - len(failures)}/{len(zips)} archives extracted into {out_root}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
