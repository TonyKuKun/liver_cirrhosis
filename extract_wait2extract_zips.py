from __future__ import annotations

import argparse
import shutil
import tarfile
from pathlib import Path, PurePosixPath
from zipfile import BadZipFile, ZipFile


DEFAULT_WAIT_DIR = Path(r"F:\PCG data\dataset\wait2extract")
DONE_MARKER = ".zip_extracted"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract zip files inside each patient folder under wait2extract."
    )
    parser.add_argument("--wait-dir", type=Path, default=DEFAULT_WAIT_DIR)
    parser.add_argument(
        "--metadata-encoding",
        default="gbk",
        help="Encoding used by zip member names when the zip lacks UTF-8 flags.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be extracted.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing extracted files and reprocess already-extracted folders.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N patient folders, useful for testing.",
    )
    return parser.parse_args()


def has_extracted_content(patient_dir: Path) -> bool:
    for child in patient_dir.iterdir():
        if child.name == "label":
            continue
        if child.name == DONE_MARKER:
            continue
        if child.suffix.lower() == ".zip":
            continue
        return True
    return False


def safe_member_path(member_name: str) -> Path | None:
    member_path = PurePosixPath(member_name.replace("\\", "/"))
    if member_path.is_absolute() or ".." in member_path.parts:
        return None
    return Path(*member_path.parts)


def extract_zip(
    zip_path: Path,
    output_dir: Path,
    *,
    metadata_encoding: str,
    overwrite: bool,
    dry_run: bool,
) -> tuple[int, int]:
    extracted = 0
    skipped = 0

    try:
        archive = ZipFile(zip_path, metadata_encoding=metadata_encoding)
    except TypeError:
        archive = ZipFile(zip_path)

    with archive:
        for info in archive.infolist():
            relative_path = safe_member_path(info.filename)
            if relative_path is None:
                skipped += 1
                print(f"  SKIP unsafe member: {info.filename}")
                continue

            target_path = output_dir / relative_path
            if info.is_dir():
                if not dry_run:
                    target_path.mkdir(parents=True, exist_ok=True)
                continue

            if target_path.exists() and not overwrite:
                skipped += 1
                continue

            extracted += 1
            if dry_run:
                continue

            target_path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target_path.open("wb") as target:
                shutil.copyfileobj(source, target)

    return extracted, skipped


def extract_tar(
    archive_path: Path,
    output_dir: Path,
    *,
    metadata_encoding: str,
    overwrite: bool,
    dry_run: bool,
) -> tuple[int, int]:
    extracted = 0
    skipped = 0

    with tarfile.open(archive_path, mode="r|*", encoding=metadata_encoding) as archive:
        for member in archive:
            relative_path = safe_member_path(member.name)
            if relative_path is None:
                skipped += 1
                print(f"  SKIP unsafe member: {member.name}")
                continue

            target_path = output_dir / relative_path
            if member.isdir():
                if not dry_run:
                    target_path.mkdir(parents=True, exist_ok=True)
                continue

            if not member.isfile():
                skipped += 1
                print(f"  SKIP non-file member: {member.name}")
                continue

            if target_path.exists() and not overwrite:
                skipped += 1
                continue

            extracted += 1
            if dry_run:
                continue

            source = archive.extractfile(member)
            if source is None:
                skipped += 1
                extracted -= 1
                continue

            target_path.parent.mkdir(parents=True, exist_ok=True)
            with source, target_path.open("wb") as target:
                shutil.copyfileobj(source, target)

    return extracted, skipped


def extract_archive(
    archive_path: Path,
    output_dir: Path,
    *,
    metadata_encoding: str,
    overwrite: bool,
    dry_run: bool,
) -> tuple[str, int, int]:
    try:
        extracted, skipped = extract_zip(
            archive_path,
            output_dir,
            metadata_encoding=metadata_encoding,
            overwrite=overwrite,
            dry_run=dry_run,
        )
        return "zip", extracted, skipped
    except BadZipFile:
        extracted, skipped = extract_tar(
            archive_path,
            output_dir,
            metadata_encoding=metadata_encoding,
            overwrite=overwrite,
            dry_run=dry_run,
        )
        return "tar", extracted, skipped


def main() -> int:
    args = parse_args()
    if not args.wait_dir.exists():
        raise SystemExit(f"wait-dir does not exist: {args.wait_dir}")

    patient_dirs = sorted(path for path in args.wait_dir.iterdir() if path.is_dir())
    if args.limit is not None:
        patient_dirs = patient_dirs[: args.limit]
    total_patients = 0
    total_zips = 0
    total_files = 0
    total_errors = 0
    skipped_patients = 0

    mode = "DRY-RUN" if args.dry_run else "EXTRACT"
    print(f"Mode: {mode}")
    print(f"Patient folders: {len(patient_dirs)}")

    for patient_dir in patient_dirs:
        zip_files = sorted(patient_dir.glob("*.zip"))
        if not zip_files:
            skipped_patients += 1
            print(f"SKIP no zip: {patient_dir.name}")
            continue

        marker = patient_dir / DONE_MARKER
        if not args.overwrite and (marker.exists() or has_extracted_content(patient_dir)):
            skipped_patients += 1
            print(f"SKIP already extracted: {patient_dir.name}")
            continue

        total_patients += 1
        patient_had_error = False
        print(f"PATIENT {patient_dir.name}")
        for zip_path in zip_files:
            total_zips += 1
            print(f"  ZIP {zip_path.name}")
            try:
                archive_type, extracted, skipped = extract_archive(
                    zip_path,
                    patient_dir,
                    metadata_encoding=args.metadata_encoding,
                    overwrite=args.overwrite,
                    dry_run=args.dry_run,
                )
            except (tarfile.TarError, OSError, EOFError) as exc:
                total_errors += 1
                patient_had_error = True
                print(f"  ERROR cannot extract archive: {exc}")
                continue
            total_files += extracted
            print(
                f"  archive_type={archive_type}, "
                f"extracted_files={extracted}, skipped_files={skipped}"
            )

        if not args.dry_run and not patient_had_error:
            marker.write_text("ok\n", encoding="utf-8")

    print(
        "Summary: "
        f"patients_extracted={total_patients}, "
        f"zip_files={total_zips}, "
        f"files_extracted={total_files}, "
        f"patients_skipped={skipped_patients}, "
        f"archive_errors={total_errors}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
