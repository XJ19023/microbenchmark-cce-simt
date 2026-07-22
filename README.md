# microbenchmark-cce-simtvf

This is a minimal CCE SIMT-VF camodel benchmark for A5 / `Ascend950PR_9599`.

Default flow:

```text
op_kernel/kernel.cce
  -> ccec
  -> build/foo_add_mix.o
  -> native runner
  -> msprof op simulator
  -> result/<timestamp>/run/*.dump and result/<timestamp>/msprof/OPPROF_*
```

## Quick Start

On this machine:

```bash
cd /mnt/d/mask_test/microbenchmark-cce-simtvf
source set_env.sh
python3 run_test.py
```

On another machine, edit or override `CANN_HOME`:

```bash
export CANN_HOME=/absolute/path/to/cann-9.0.0
export ARCH=x86_64-linux        # or aarch64-linux
cd /path/to/microbenchmark-cce-simtvf
source set_env.sh
python3 run_test.py
```

## What To Modify

Only edit:

```text
op_kernel/kernel.cce
```

Declare the kernel symbol and every GM tensor argument in comments near the top
of the same file:

```cpp
// MB_KERNEL foo_add
// MB_TENSOR output A int32 256
// MB_TENSOR input B int32 256
// MB_TENSOR input C int32 256
```

The `MB_TENSOR` lines must follow the exact order of the kernel function
parameters. `run_test.py` reads these declarations, allocates every tensor,
builds the kernel argument list, initializes inputs, launches the kernel, and
writes every output back to a `.bin` file. The number, dtype, direction, and
length of tensors are not hard-coded in the runner.

Syntax:

```text
// MB_KERNEL <kernel_symbol>
// MB_TENSOR <input|output|inout> <parameter_name> <dtype> <element_count_or_shape>
```

Examples of valid lengths and shapes:

```cpp
// MB_TENSOR input X fp32 1024
// MB_TENSOR output Y fp32 8x1024
// MB_TENSOR inout State uint32 TOTAL_COUNT
```

`8x1024` is flattened to 8192 elements. `TOTAL_COUNT` uses the value of
`--total-count`, whose default is 256. Supported dtypes are `int8`, `uint8`,
`int16`, `uint16`, `int32`, `uint32`, `int64`, `uint64`, `fp16`, `bf16`,
`fp32`, and `fp64`.

For example, a kernel with two outputs and four inputs only needs six
`MB_TENSOR` lines followed by the matching six pointer parameters. No Python or
runner source needs to be changed. A complete example is provided in
`op_kernel/kernel_multi_io.cce`.

An AscendC SIMT-wrapper example is provided at:

```text
op_kernel/kernel_ascendc_exp.cce
```

Run it with:

```bash
python3 run_test.py --kernel op_kernel/kernel_ascendc_exp.cce
```

The generic runner currently supports GM tensor pointer parameters. Kernel
scalar parameters require ABI-aware packing and are intentionally not inferred.
Use compile-time constants in the kernel for a tensor-only microbenchmark.

The old fixed `A/B/C` runner remains available for CCE files without any
`MB_KERNEL` or `MB_TENSOR` declarations. Its `--dtype` and `--golden` options
continue to work in that legacy mode. Generic tensor mode saves outputs but
does not perform a golden comparison.

## Useful Commands

Run without `msprof`, useful for faster functional checks:

```bash
python3 run_test.py --no-msprof
```

Run the included float exp kernel:

```bash
python3 run_test.py --kernel op_kernel/kernel_ascendc_exp.cce
```

Use a fixed output directory:

```bash
python3 run_test.py --output result/manual_run
```

Change total elements:

```bash
python3 run_test.py --total-count 1024
```

The script passes `-DTOTAL_COUNT=<value>` to `ccec`, and `TOTAL_COUNT` in the
tensor declaration follows the same value. The kernel must still launch or loop
over enough SIMT threads to cover that many elements.

Set dynamic UB / local memory size for SIMT launch:

```bash
python3 run_test.py --local-memory-size 65536
```

The default is `0`, which keeps the runtime default:

```bash
export LOCAL_MEMORY_SIZE=0
python3 run_test.py
```

In CANN 9.0, `LocalMemorySize` and `DYN_UBUF_SIZE` identify the same launch
attribute. The runner uses the older `LOCAL_MEMORY_SIZE` spelling because it is
available in both CANN 9.0 beta.1 and the formal CANN 9.0.0 package.

Compile only, without starting camodel:

```bash
python3 run_test.py --compile-only
```

## Output

Each run creates:

```text
result/<timestamp>/build/foo_add_mix_aiv.o
result/<timestamp>/build/foo_add_mix.o
result/<timestamp>/build/native_cce_runner
result/<timestamp>/build/kernel_tensors.tsv
result/<timestamp>/run/input_<name>.bin
result/<timestamp>/run/output_<name>.bin
result/<timestamp>/run/*.dump
result/<timestamp>/msprof/OPPROF_*
result/<timestamp>/run.log
```

`run.log` contains all compile and runtime commands.

`msprof op simulator` writes profiling data to a private `/tmp` directory first,
then `run_test.py` copies the generated `OPPROF_*` directory back to
`result/<timestamp>/msprof/`. This avoids permission checks on mounted paths such
as `/mnt/d`.

If the run log contains `CreateSocket ... Operation not permitted` and the
kernel passes but no dump files are generated, the process is being restricted by
the shell/container sandbox. Run the same command in a normal terminal session.
The expected successful log contains:

```text
[INFO] <ProfInit> Start profiling on kernel: foo_add
[OUTPUT] A dtype=int32 elements=256 file=output_A.bin
Profiling data parse finished.
```

Useful SIMT dump files include:

```text
msprof/OPPROF_*/dump/core0.veccore0.instr_popped_log.dump
msprof/OPPROF_*/dump/core0.veccore0.rvec.simt.*.dump
msprof/OPPROF_*/simulator/trace.json
msprof/OPPROF_*/simulator/visualize_data.bin
```

## SIMT Indexing

The default `op_kernel/kernel.cce` is the pure CCE form. It uses:

```cpp
cce::async_invoke<callee>(cce::dim3{256, 1, 1}, A, B, C);
```

Use one of these forms inside `__simt_vf__`:

```cpp
for (int tid = __cce_simt_get_TID_X(); tid < TOTAL_COUNT;
     tid += __cce_simt_get_BLOCK_DIM_X()) {
    A[tid] = B[tid] + C[tid];
}
```

or, if using the AscendC SIMT wrapper:

```cpp
for (int tid = static_cast<int>(Simt::GetThreadIdx()); tid < TOTAL_COUNT;
     tid += static_cast<int>(Simt::GetThreadNum())) {
    A[tid] = B[tid] + C[tid];
}
```

Avoid `threadIdx.x` for this CCE SIMT-VF path.
