"""
Copy OOD plots from nested results/ood/plots/ structure into a flat consolidated folder:
  results/ood/plots_consolidated_ood/
    linear_hetero/
    linear_homo/
    sin_hetero/
    sin_homo/
Existing plot structure is left unchanged (copy only).
"""
from pathlib import Path
import shutil
import re


def get_consolidated_folder_name(func_type: str, noise_type: str) -> str:
    """Map (func_type, noise_type) to folder name: linear_hetero, linear_homo, sin_hetero, sin_homo."""
    short = "hetero" if noise_type == "heteroscedastic" else "homo"
    return f"{func_type}_{short}"


def sanitize_filename(name: str) -> str:
    """Replace characters that are invalid on Windows."""
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    return name[:200]  # avoid very long paths


def rename_beta_to_ascii_in_consolidated(ood_root: Path) -> int:
    """
    Rename PNGs in plots_consolidated_ood so that Unicode β (U+03B2) becomes ASCII 'beta'.
    Returns the number of files renamed. Skips if target name already exists (different file).
    """
    out_base = ood_root / "plots_consolidated_ood"
    folders = ["linear_hetero", "linear_homo", "sin_hetero", "sin_homo"]
    beta_char = "\u03b2"
    renamed_count = 0
    for folder in folders:
        dest_dir = out_base / folder
        if not dest_dir.is_dir():
            continue
        for png in dest_dir.glob("*.png"):
            if beta_char not in png.name:
                continue
            new_name = png.name.replace(beta_char, "beta")
            if new_name == png.name:
                continue
            dest_file = dest_dir / new_name
            if dest_file.exists() and dest_file.resolve() != png.resolve():
                continue
            png.rename(dest_file)
            renamed_count += 1
    return renamed_count


def consolidate_ood_plots(ood_root: Path) -> int:
    """
    Copy all PNGs under ood_root/plots into ood_root/plots_consolidated_ood/{linear_hetero,linear_homo,sin_hetero,sin_homo}.
    Expects source paths like: .../plots/uncertainties_ood/heteroscedastic/linear/... or .../homoscedastic/sin/...
    """
    plots_dir = ood_root / "plots"
    out_base = ood_root / "plots_consolidated_ood"
    folders = ["linear_hetero", "linear_homo", "sin_hetero", "sin_homo"]
    for f in folders:
        (out_base / f).mkdir(parents=True, exist_ok=True)

    count = 0
    ood_subdirs = [
        "uncertainties_ood",
        "uncertainties_entropy_ood",
        "uncertainties_entropy_lines_ood",
    ]
    if not plots_dir.exists():
        return 0

    for subdir_name in ood_subdirs:
        subdir = plots_dir / subdir_name
        if not subdir.is_dir():
            continue
        for noise_type in ("heteroscedastic", "homoscedastic"):
            for func_type in ("linear", "sin"):
                src_dir = subdir / noise_type / func_type
                if not src_dir.is_dir():
                    continue
                folder_name = get_consolidated_folder_name(func_type, noise_type)
                dest_dir = out_base / folder_name
                for png in src_dir.glob("*.png"):
                    safe_name = sanitize_filename(png.name).replace("\u03b2", "beta")
                    dest_file = dest_dir / safe_name
                    if dest_file.exists() and dest_file.stat().st_size != png.stat().st_size:
                        prefix = subdir_name.replace("uncertainties_", "").replace("_ood", "") or "ood"
                        safe_name = sanitize_filename(f"{prefix}_{png.name}").replace("\u03b2", "beta")
                        dest_file = dest_dir / safe_name
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    data = png.read_bytes()
                    dest_file.write_bytes(data)
                    count += 1
    return count


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    ood_root = project_root / "results" / "ood"
    n = consolidate_ood_plots(ood_root)
    print(f"Copied {n} plots to {ood_root / 'plots_consolidated_ood'}")
    r = rename_beta_to_ascii_in_consolidated(ood_root)
    print(f"Renamed {r} filenames (beta to ASCII) in plots_consolidated_ood")
