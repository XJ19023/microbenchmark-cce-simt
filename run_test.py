#!/usr/bin/env python3
import argparse
import datetime as dt
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent

SUPPORTED_TENSOR_DTYPES = {
    "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64",
    "fp16", "bf16", "fp32", "float32", "fp64", "float64",
}


@dataclass(frozen=True)
class TensorSpec:
    direction: str
    name: str
    dtype: str
    elements: int


@dataclass(frozen=True)
class KernelManifest:
    kernel_name: str
    tensors: tuple[TensorSpec, ...]


def parse_kernel_manifest(kernel_path: Path, total_count: int) -> KernelManifest | None:
    """Parse MB_KERNEL/MB_TENSOR declarations embedded in a CCE source file."""
    text = kernel_path.read_text(encoding="utf-8")
    kernel_matches = re.findall(r"^\s*//\s*MB_KERNEL\s+([A-Za-z_]\w*)\s*$", text, re.MULTILINE)
    tensor_matches = re.findall(
        r"^\s*//\s*MB_TENSOR\s+(input|output|inout)\s+([A-Za-z_]\w*)\s+"
        r"([A-Za-z0-9_]+)\s+([A-Za-z0-9_xX]+)\s*$",
        text,
        re.MULTILINE,
    )
    if not kernel_matches and not tensor_matches:
        return None
    if len(kernel_matches) != 1:
        raise RuntimeError("generic mode requires exactly one '// MB_KERNEL <symbol>' declaration")
    if not tensor_matches:
        raise RuntimeError("generic mode requires at least one '// MB_TENSOR ...' declaration")

    tensors = []
    names = set()
    for direction, name, dtype, shape in tensor_matches:
        dtype = {"float32": "fp32", "float64": "fp64"}.get(dtype, dtype)
        if dtype not in SUPPORTED_TENSOR_DTYPES:
            supported = ", ".join(sorted(SUPPORTED_TENSOR_DTYPES))
            raise RuntimeError(f"unsupported dtype '{dtype}' in MB_TENSOR; supported: {supported}")
        if name in names:
            raise RuntimeError(f"duplicate MB_TENSOR name: {name}")
        names.add(name)

        if shape == "TOTAL_COUNT":
            elements = total_count
        else:
            dims = re.split(r"[xX]", shape)
            if not dims or any(not dim.isdigit() or int(dim) <= 0 for dim in dims):
                raise RuntimeError(
                    f"invalid shape '{shape}' for tensor {name}; use a positive count, e.g. 256 or 8x1024"
                )
            elements = 1
            for dim in dims:
                elements *= int(dim)
        tensors.append(TensorSpec(direction, name, dtype, elements))

    return KernelManifest(kernel_matches[0], tuple(tensors))


def write_tensor_spec(manifest: KernelManifest, path: Path) -> None:
    lines = [f"{item.direction}\t{item.name}\t{item.dtype}\t{item.elements}\n" for item in manifest.tensors]
    path.write_text("".join(lines), encoding="utf-8")


def default_cann_home() -> str:
    env = os.environ.get("CANN_HOME")
    if env:
        return env
    local = Path("/home/lenovo/.codex/memories/cann-9.0.0/cann-9.0.0")
    if local.exists():
        return str(local)
    return ""


def default_arch() -> str:
    env = os.environ.get("ARCH")
    if env:
        return env
    machine = platform.machine()
    if machine == "x86_64":
        return "x86_64-linux"
    if machine in ("aarch64", "arm64"):
        return "aarch64-linux"
    return f"{machine}-linux"


def timestamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def quote(value) -> str:
    return "'" + str(value).replace("'", "'\"'\"'") + "'"


def run_bash(script: str, cwd: Path, log_path: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        ["bash", "-lc", script],
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=os.environ.copy(),
    )
    if log_path:
        with log_path.open("a", encoding="utf-8") as f:
            f.write(proc.stdout)
    if proc.stdout:
        print(proc.stdout, end="")
    if check and proc.returncode != 0:
        raise RuntimeError(f"command failed with code {proc.returncode}")
    return proc


def require_file(path: Path, what: str) -> Path:
    if not path.exists():
        raise RuntimeError(f"{what} not found: {path}")
    return path


def cce_include_flags(cann_home: Path, arch: str) -> list[str]:
    candidates = [
        Path("/usr/include/c++/11"),
        Path("/usr/include/c++/12"),
        Path("/usr/include/c++/13"),
        Path("/usr/include/x86_64-linux-gnu/c++/11"),
        Path("/usr/include/x86_64-linux-gnu/c++/12"),
        Path("/usr/include/x86_64-linux-gnu/c++/13"),
        Path("/usr/include/aarch64-linux-gnu/c++/11"),
        Path("/usr/include/aarch64-linux-gnu/c++/12"),
        Path("/usr/include/aarch64-linux-gnu/c++/13"),
        cann_home / "include",
        cann_home / arch / "include",
        cann_home / arch / "asc",
        cann_home / arch / "asc" / "include",
        cann_home / arch / "asc" / "include" / "basic_api",
        cann_home / arch / "asc" / "include" / "interface",
        cann_home / arch / "asc" / "impl",
        cann_home / arch / "asc" / "impl" / "basic_api",
        cann_home / arch / "pkg_inc",
    ]
    flags = []
    for item in candidates:
        if item.is_dir():
            flags.append(f"-I{item}")
    return flags


def find_tool(cann_home: Path, arch: str, name: str) -> str:
    candidates = [
        cann_home / arch / "bin" / name,
        cann_home / "bin" / name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    found = shutil.which(name)
    if found:
        return found
    raise RuntimeError(f"{name} not found. Source set_env.sh or check CANN_HOME/ARCH.")


def compile_kernel(args, cann_home: Path, arch: str, build_dir: Path, log_path: Path) -> Path:
    kernel_src = args.kernel.resolve()
    require_file(kernel_src, "CCE kernel")
    ccec = find_tool(cann_home, arch, "ccec")
    ld_lld = find_tool(cann_home, arch, "ld.lld")
    aiv_obj = build_dir / f"{args.kernel_name}_mix_aiv.o"
    kernel_bin = build_dir / f"{args.kernel_name}_mix.o"
    includes = " ".join(quote(flag) for flag in cce_include_flags(cann_home, arch))
    extra_flags = os.environ.get("CCEC_EXTRA_FLAGS", "")
    script = f"""
set -e
source {quote(cann_home / 'set_env.sh')}
export ASCEND_HOME_PATH={quote(cann_home)}
export ASCEND_CANN_PACKAGE_PATH={quote(cann_home)}
{quote(ccec)} -g -std=c++17 -c -O2 {quote(kernel_src)} -o {quote(aiv_obj)} \
  {includes} \
  --cce-aicore-arch={quote(args.core_arch)} \
  --cce-aicore-only \
  -mllvm -cce-aicore-function-stack-size=16000 \
  -mllvm -cce-aicore-record-overflow=false \
  -mllvm -cce-aicore-addr-transform \
  -mllvm -cce-aicore-jump-expand=true \
  --cce-simd-vf-fusion=false \
  -DTOTAL_COUNT={args.total_count} \
  {extra_flags}
{quote(ld_lld)} -Ttext=0 {quote(aiv_obj)} -static -o {quote(kernel_bin)}
"""
    run_bash(script, cwd=REPO_ROOT, log_path=log_path)
    return kernel_bin


def compile_runner(args, cann_home: Path, arch: str, build_dir: Path, log_path: Path,
                   generic_mode: bool) -> Path:
    runner_name = "generic_runner.cpp" if generic_mode else "native_runner.cpp"
    runner_src = REPO_ROOT / "runner" / runner_name
    runner_bin = build_dir / "native_cce_runner"
    host_inc = cann_home / arch / "include"
    pkg_inc = cann_home / arch / "pkg_inc"
    cann_lib = cann_home / arch / "lib64"
    sim_lib = cann_home / arch / "simulator" / args.soc_version / "lib"
    dav_lib = cann_home / arch / "simulator" / "dav_3510" / "lib"
    dav_camodel = cann_home / arch / "simulator" / "dav_3510" / "camodel"
    devlib = cann_home / arch / "devlib"
    devlib_device = cann_home / arch / "devlib" / "device"
    device_lib = cann_home / arch / "lib64" / "device" / "lib64"
    script = f"""
set -e
source {quote(cann_home / 'set_env.sh')}
g++ -std=c++17 -O2 -Wl,-z,relro -Wl,-z,now -Wl,--allow-shlib-undefined \
  -o {quote(runner_bin)} {quote(runner_src)} \
  -I{quote(host_inc)} \
  -I{quote(host_inc / 'experiment' / 'msprof')} \
  -I{quote(host_inc / 'experiment' / 'msprof' / 'toolchain')} \
  -I{quote(cann_home / 'include')} \
  -I{quote(pkg_inc)} \
  -I{quote(pkg_inc / 'runtime')} \
  -I{quote(pkg_inc / 'profiling')} \
  -I{quote(cann_home / arch / 'pkg_inc' / 'toolchain')} \
  -L{quote(cann_lib)} \
  -L{quote(sim_lib)} \
  -L{quote(dav_lib)} \
  -L{quote(dav_camodel)} \
  -L{quote(devlib)} \
  -Wl,-rpath,{quote(str(cann_lib) + ':' + str(sim_lib) + ':' + str(dav_lib) + ':' + str(dav_camodel) + ':' + str(devlib) + ':' + str(devlib_device) + ':' + str(device_lib) + ':' + str(build_dir))} \
  -lruntime_camodel -lstdc++ -lascendcl -lm -ltiling_api -lplatform -lc_sec -ldl -lnnopbase
"""
    run_bash(script, cwd=REPO_ROOT, log_path=log_path)
    return runner_bin


def runtime_env_script(cann_home: Path, arch: str, args, run_dir: Path) -> str:
    host_machine = platform.machine()
    host_devlib = "aarch64" if host_machine in ("aarch64", "arm64") else "x86_64"
    other_devlib = "x86_64" if host_devlib == "aarch64" else "aarch64"
    paths = [
        cann_home / "lib64",
        cann_home / "fwkacllib" / "lib64",
        cann_home / "runtime" / "lib64",
        cann_home / arch / "lib64",
        cann_home / arch / "lib64" / "device" / "lib64",
        cann_home / arch / "devlib",
        cann_home / arch / "devlib" / "device",
        cann_home / arch / "devlib" / "linux" / host_devlib,
        cann_home / arch / "devlib" / "linux" / other_devlib,
        cann_home / arch / "simulator" / "dav_3510" / "camodel",
        cann_home / arch / "simulator" / "dav_3510" / "lib",
        cann_home / "tools" / "simulator" / args.soc_version / "lib",
        cann_home / arch / "simulator" / args.soc_version / "lib",
    ]
    ld = ":".join(str(p) for p in paths if p.exists())
    return f"""
source {quote(cann_home / 'set_env.sh')}
export ASCEND_HOME_PATH={quote(cann_home)}
export ASCEND_CANN_PACKAGE_PATH={quote(cann_home)}
export ASCEND_TOOLKIT_HOME={quote(cann_home)}
export ASCEND_OPP_PATH={quote(cann_home / 'opp')}
export ASCEND_DEVICE_ID={args.device_id}
export ACL_DEVICE_ID={args.device_id}
export ASCEND_PROCESS_LOG_PATH={quote(run_dir / 'ascend_process_log')}
mkdir -p "$ASCEND_PROCESS_LOG_PATH"
export LD_LIBRARY_PATH={quote(ld)}:${{LD_LIBRARY_PATH:-}}
unset LD_PRELOAD
"""


def copy_camodel_config(cann_home: Path, arch: str, args, run_dir: Path) -> None:
    etc = run_dir / "etc"
    etc.mkdir(parents=True, exist_ok=True)
    candidates = []
    env_cfg = os.environ.get("CAMODEL_CONFIG_TOML")
    if env_cfg:
        candidates.append(Path(env_cfg))
    candidates += [
        cann_home / arch / "simulator" / "dav_3510" / "lib" / "1982_cloud_config.toml",
        cann_home / arch / "simulator" / args.soc_version / "lib" / "1982_cloud_config.toml",
        cann_home / "tools" / "simulator" / args.soc_version / "lib" / "1982_cloud_config.toml",
    ]
    for candidate in candidates:
        if candidate.exists():
            shutil.copy2(candidate, etc / "1982_cloud_config.toml")
            print(f"[INFO] copied camodel config: {candidate}")
            return
    print("[WARN] 1982_cloud_config.toml not found; set CAMODEL_CONFIG_TOML if core_wrapper needs it")


def run_kernel(args, cann_home: Path, arch: str, out_dir: Path, kernel_bin: Path, runner_bin: Path,
               log_path: Path, tensor_spec: Path | None) -> None:
    run_dir = out_dir / "run"
    msprof_dir = out_dir / "msprof"
    msprof_work_dir = Path(tempfile.mkdtemp(prefix="cce_msprof_"))
    run_dir.mkdir(parents=True, exist_ok=True)
    msprof_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "log_ca").mkdir(parents=True, exist_ok=True)
    for item in (out_dir, run_dir, msprof_dir, msprof_work_dir):
        try:
            item.chmod(0o700)
        except PermissionError:
            pass
    shutil.copy2(kernel_bin, run_dir / kernel_bin.name)
    shutil.copy2(runner_bin, run_dir / runner_bin.name)
    if tensor_spec is not None:
        shutil.copy2(tensor_spec, run_dir / tensor_spec.name)
    os.chmod(run_dir / runner_bin.name, 0o755)
    copy_camodel_config(cann_home, arch, args, run_dir)

    env_script = runtime_env_script(cann_home, arch, args, run_dir)
    if tensor_spec is not None:
        app = (
            f"./{runner_bin.name} ./{kernel_bin.name} {args.kernel_name} ./{tensor_spec.name} "
            f"{args.block_dim} {args.local_memory_size}"
        )
    else:
        app = (
            f"./{runner_bin.name} ./{kernel_bin.name} {args.kernel_name} {args.dtype} "
            f"{args.total_count} {args.block_dim} {args.local_memory_size} {args.golden} {args.atol} {args.rtol}"
        )
    if args.no_msprof:
        command = f"""
set -e
{env_script}
cd {quote(run_dir)}
{app}
"""
    else:
        command = f"""
set -e
{env_script}
cd {quote(run_dir)}
msprof op simulator \
  --soc-version={quote(args.soc_version)} \
  --kernel-name={quote(args.kernel_name)} \
  --timeout={args.timeout} \
  --output={quote(msprof_work_dir)} \
  {app}
"""
    run_bash(command, cwd=REPO_ROOT, log_path=log_path)
    for item in msprof_work_dir.iterdir():
        target = msprof_dir / item.name
        if target.exists():
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)
    shutil.rmtree(msprof_work_dir, ignore_errors=True)


def summarize(out_dir: Path) -> None:
    print(f"[INFO] OUTPUT={out_dir}")
    for path in [
        out_dir / "build",
        out_dir / "run",
        out_dir / "msprof",
    ]:
        if path.exists():
            print(f"[INFO] {path.name}={path}")
    dump_files = sorted((out_dir / "run").glob("*.dump")) if (out_dir / "run").exists() else []
    if dump_files:
        print("[INFO] run dump samples:")
        for item in dump_files[:20]:
            print(f"  {item.name} {item.stat().st_size} bytes")
    opprofs = sorted((out_dir / "msprof").glob("OPPROF_*")) if (out_dir / "msprof").exists() else []
    if opprofs:
        latest = opprofs[-1]
        print(f"[INFO] latest OPPROF={latest}")
        interesting = []
        for pattern in ("**/*.dump", "**/trace.json", "**/visualize_data.bin"):
            interesting.extend(latest.glob(pattern))
        for item in sorted(interesting)[:40]:
            print(f"  {item.relative_to(latest)} {item.stat().st_size} bytes")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compile and run a CCE SIMT-VF microbenchmark on A5 camodel.")
    parser.add_argument("--cann-home", default=default_cann_home())
    parser.add_argument("--arch", default=default_arch())
    parser.add_argument("--soc-version", default=os.environ.get("SOC_VERSION", "Ascend950PR_9599"))
    parser.add_argument("--core-arch", default=os.environ.get("CORE_ARCH", "dav-c310-vec"))
    parser.add_argument("--kernel", type=Path, default=REPO_ROOT / "op_kernel" / "kernel.cce")
    parser.add_argument("--kernel-name", default="foo_add")
    parser.add_argument("--dtype", choices=("int32", "fp32", "float32"), default="int32")
    parser.add_argument("--total-count", type=int, default=256)
    parser.add_argument("--block-dim", type=int, default=1)
    parser.add_argument("--local-memory-size", type=int, default=int(os.environ.get("LOCAL_MEMORY_SIZE", "0")),
                        help="Dynamic UB/local memory size in bytes. 0 keeps the runtime default.")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--golden", choices=("none", "add", "exp_mul"), default="add")
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--timeout", type=int, default=10)
    parser.add_argument("--no-msprof", action="store_true", help="Run the native runner directly without msprof.")
    parser.add_argument("--compile-only", action="store_true", help="Only compile the kernel and native runner.")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    cann_home = Path(args.cann_home).expanduser().resolve()
    if not cann_home.exists():
        raise RuntimeError(f"CANN_HOME/--cann-home does not exist: {cann_home}")
    if args.total_count <= 0:
        raise RuntimeError("--total-count must be positive")
    if args.block_dim <= 0:
        raise RuntimeError("--block-dim must be positive")
    if args.local_memory_size < 0:
        raise RuntimeError("--local-memory-size must be non-negative")

    out_dir = (args.output or (REPO_ROOT / "result" / timestamp())).resolve()
    build_dir = out_dir / "build"
    out_dir.mkdir(parents=True, exist_ok=True)
    build_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"

    print(f"CANN_HOME={cann_home}")
    print(f"ARCH={args.arch}")
    print(f"OUTPUT={out_dir}")

    kernel_path = args.kernel.expanduser().resolve()
    require_file(kernel_path, "CCE kernel")
    manifest = parse_kernel_manifest(kernel_path, args.total_count)
    tensor_spec = None
    if manifest is not None:
        args.kernel_name = manifest.kernel_name
        tensor_spec = build_dir / "kernel_tensors.tsv"
        write_tensor_spec(manifest, tensor_spec)
        print(f"[INFO] generic tensor mode: kernel={manifest.kernel_name}, args={len(manifest.tensors)}")
        for index, item in enumerate(manifest.tensors):
            print(f"  arg[{index}] {item.direction} {item.name}: {item.dtype}[{item.elements}]")
        if args.golden != "none":
            print("[INFO] generic tensor mode does not run a built-in golden check")
    else:
        print("[INFO] legacy three-tensor mode (no MB_KERNEL/MB_TENSOR declarations found)")

    kernel_bin = compile_kernel(args, cann_home, args.arch, build_dir, log_path)
    runner_bin = compile_runner(args, cann_home, args.arch, build_dir, log_path, manifest is not None)
    if args.compile_only:
        print(f"[INFO] kernel_bin={kernel_bin}")
        print(f"[INFO] runner_bin={runner_bin}")
        summarize(out_dir)
        return 0
    run_kernel(args, cann_home, args.arch, out_dir, kernel_bin, runner_bin, log_path, tensor_spec)
    summarize(out_dir)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
