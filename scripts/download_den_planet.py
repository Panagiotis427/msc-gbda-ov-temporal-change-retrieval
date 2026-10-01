"""
Download + prepare the *native* Planet-Fusion DynamicEarthNet (official TUM zips).

This is the full-resolution counterpart to ``scripts/download_den.py`` (the ~7 GB
preprocessed gdown subset). It cherry-picks only the UTM-zone archives you ask for
from the TUM dataserv rsync mirror — instead of the full ~525 GB release — and
verifies them through the :mod:`src.datasets.dynamic_earthnet_planet` loader.

Usage::

    # default zones reproduce the report's 23-AOI experiment (all 7 classes)
    python -m scripts.download_den_planet --dest data/dynamic_earthnet_planet

    # a specific subset
    python -m scripts.download_den_planet --dest <dir> --zones 36N 19S 32N

    # already downloaded elsewhere (e.g. an external drive) — just verify
    python -m scripts.download_den_planet --dest /media/<drive>/dynamic_earthnet --skip-download

Steps
-----
1. ``rsync`` the requested ``planet.<zone>.zip`` archives + ``labels.zip`` +
   ``checksums.sha512`` (idempotent — ``-P`` resumes/repairs partial files;
   skips ``_done.marker``).
2. Optional ``--verify`` — ``sha512sum -c`` the downloaded zips against the
   manifest.
3. Verify the layout by building the loader index (counts usable cubes =
   imagery ∩ raster-label, per zone).
4. Touch ``_done.marker``.

Notes
-----
- ``rsync`` must be installed; this script shells out to it and never writes to
  the source mirror.
- The labels archive (``labels.zip``, all 55 cubes) is small and always fetched,
  so the loader can see every zone's class coverage even before its imagery is
  downloaded.
- To pick which zones add the rarest classes (agriculture / wetlands / snow),
  check the labels' class coverage per zone first; snow lives only in 19S & 32N,
  wetlands only in 36N / 21S / 33N.
"""
import argparse
import os
import subprocess
from pathlib import Path

# TUM dataserv rsync module for the official DynamicEarthNet (Planet Fusion) release.
_RSYNC_URL = "rsync://m1650201@dataserv.ub.tum.de/m1650201/"

# Zones whose imagery reproduces the report's 23-AOI run (spans all 7 classes).
DEFAULT_ZONES = (
    "10N", "11N", "13N", "15N", "16N", "17N", "18N",   # original 8-cube set
    "19S",                                              # + snow + agriculture
    "36N", "21S", "33N",                                # + wetlands
    "32N",                                              # + snow into the test split
)


def _rsync_download(dest: Path, zones, rsync_url: str) -> None:
    """rsync only ``labels.zip`` + the requested ``planet.<zone>.zip`` + the manifest.

    ``--include`` rules are evaluated before the trailing ``--exclude='*'``, so only
    the named files transfer; ``-P`` resumes partial files (which also repairs a
    previously-corrupt archive for free).
    """
    includes = ["--include=labels.zip", "--include=checksums.sha512"]
    includes += [f"--include=planet.{z}.zip" for z in zones]
    cmd = ["rsync", "-avP", *includes, "--exclude=*", rsync_url, str(dest) + os.sep]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def verify_checksums(dest: Path, zones) -> None:
    """``sha512sum -c`` the downloaded zips against ``checksums.sha512``."""
    manifest = dest / "checksums.sha512"
    if not manifest.exists():
        print(f"No checksums.sha512 at {dest} — skipping checksum verify.")
        return
    targets = {f"planet.{z}.zip" for z in zones} | {"labels.zip"}
    lines = [
        ln for ln in manifest.read_text().splitlines()
        if any(t in ln for t in targets)
    ]
    if not lines:
        print("No matching entries in checksums.sha512 — skipping.")
        return
    print(f"Verifying {len(lines)} checksum(s) ...")
    subprocess.run(
        ["sha512sum", "-c", "-"], input="\n".join(lines) + "\n",
        text=True, cwd=str(dest), check=True,
    )
    print("Checksums OK.")


def verify_layout(dest: Path) -> int:
    """Build the loader index and report usable cubes (imagery ∩ raster-label) per zone."""
    from src.datasets.dynamic_earthnet_planet import build_index

    index = build_index(str(dest))
    total = 0
    for zone in sorted(index):
        cubes = sorted(index[zone])
        total += len(cubes)
        print(f"  {zone}: {len(cubes)} usable cube(s) — {', '.join(cubes)}")
    if total == 0:
        raise RuntimeError(
            f"No usable cubes found under {dest}. Check that labels.zip and at least "
            "one planet.<zone>.zip are present and intact."
        )
    print(f"Layout OK — {total} usable cubes across {len(index)} zone(s).")
    return total


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download + prepare native 3 m Planet DynamicEarthNet (TUM zips)")
    parser.add_argument("--dest", type=str, default="data/dynamic_earthnet_planet",
                        help="Directory to hold the zips (labels.zip + planet.<zone>.zip)")
    parser.add_argument("--zones", type=str, nargs="+", default=list(DEFAULT_ZONES),
                        help="UTM zones to fetch (e.g. 36N 19S 32N). Default: the 23-AOI set.")
    parser.add_argument("--rsync-url", type=str, default=_RSYNC_URL,
                        help="TUM dataserv rsync module URL")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip rsync (zips already present) — verify only")
    parser.add_argument("--verify", action="store_true",
                        help="sha512sum -c the downloaded zips against checksums.sha512")
    args = parser.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    marker = dest / "_done.marker"

    if marker.exists():
        print(f"Planet DEN already prepared at {dest} (found _done.marker). Nothing to do.")
        return

    if not args.skip_download:
        _rsync_download(dest, args.zones, args.rsync_url)
    else:
        print(f"--skip-download set — verifying existing files under {dest}")

    if args.verify:
        verify_checksums(dest, args.zones)

    verify_layout(dest)

    marker.touch()
    print(f"\nPlanet DEN preparation complete. Data at: {dest}")
    print("Next: build embeddings with the 'dynamic_earthnet_planet' dataset, e.g.\n"
          f"  python -m scripts.run_pipeline --dataset dynamic_earthnet_planet --root {dest}")


if __name__ == "__main__":
    main()
