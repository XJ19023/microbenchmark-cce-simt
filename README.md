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

For the simplest workflow, only edit:

```text
op_kernel/kernel.cce
```

Keep this ABI if you want to reuse the default runner unchanged:

```cpp
extern "C" __global__ [aicore]
void foo_add(__gm__ T *A, __gm__ T *B, __gm__ T *C)
```

`A` is output, `B/C` are inputs. The runner allocates all three buffers, launches with `block_dim=1`, copies `A` back, and optionally checks a golden result.

If you change the kernel symbol name, pass it to the runner:

```bash
python3 run_test.py --kernel-name your_kernel_name
```

If you change dtype to `float`, also update the CCE pointer types and run:

```bash
python3 run_test.py --dtype fp32 --golden exp_mul
```

An AscendC SIMT-wrapper example is provided at:

```text
op_kernel/kernel_ascendc_exp.cce
```

Run it with:

```bash
python3 run_test.py --kernel op_kernel/kernel_ascendc_exp.cce --dtype fp32 --golden exp_mul
```

Use `--golden none` for kernels whose result cannot be checked by the built-in `add` or `exp_mul` golden.

## Useful Commands

Run without `msprof`, useful for faster functional checks:

```bash
python3 run_test.py --no-msprof
```

Run a float exp kernel:

```bash
python3 run_test.py --dtype fp32 --golden exp_mul --kernel op_kernel/kernel.cce
```

Use a fixed output directory:

```bash
python3 run_test.py --output result/manual_run
```

Change total elements:

```bash
python3 run_test.py --total-count 1024
```

The script passes `-DTOTAL_COUNT=<value>` to `ccec`, so the default kernel loop bound follows `--total-count`.

Set dynamic UB / local memory size for SIMT launch:

```bash
python3 run_test.py --local-memory-size 65536
```

The default is `0`, which keeps the runtime default:

```bash
export LOCAL_MEMORY_SIZE=0
python3 run_test.py
```

In CANN 9.0, `LocalMemorySize` is a deprecated name for the same launch
attribute as `DYN_UBUF_SIZE`; the runner uses `ACL_RT_LAUNCH_KERNEL_ATTR_DYN_UBUF_SIZE`.

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
result/<timestamp>/run/output0.bin
result/<timestamp>/run/input_b.bin
result/<timestamp>/run/input_c.bin
result/<timestamp>/run/golden.bin
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
[CHECK] PASS
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
