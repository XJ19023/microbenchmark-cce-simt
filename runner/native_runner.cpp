#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "runtime/kernel.h"
#include "runtime/rt.h"

namespace {

bool ReadBinaryFile(const std::string& path, std::vector<unsigned char>& buffer)
{
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::fprintf(stderr, "[ERROR] failed to open kernel binary: %s\n", path.c_str());
        return false;
    }
    file.seekg(0, std::ios::end);
    const std::streamoff size = file.tellg();
    file.seekg(0, std::ios::beg);
    if (size <= 0) {
        std::fprintf(stderr, "[ERROR] empty kernel binary: %s\n", path.c_str());
        return false;
    }
    buffer.resize(static_cast<size_t>(size));
    file.read(reinterpret_cast<char*>(buffer.data()), size);
    return static_cast<bool>(file);
}

bool WriteFile(const std::string& path, const void* data, size_t bytes)
{
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::fprintf(stderr, "[WARN] failed to write file: %s\n", path.c_str());
        return false;
    }
    file.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    return static_cast<bool>(file);
}

void PrintRecentAclError()
{
    const char* recent = aclGetRecentErrMsg();
    if (recent != nullptr && recent[0] != '\0') {
        std::fprintf(stderr, "[ACL] RecentErrMsg: %s\n", recent);
    }
}

#define ACL_CHECK(expr)                                                                          \
    do {                                                                                         \
        const aclError _ret = (expr);                                                            \
        if (_ret != ACL_SUCCESS) {                                                               \
            std::fprintf(stderr, "[ERROR] %s failed: %d (%s:%d)\n", #expr, (int)_ret, __FILE__, \
                         __LINE__);                                                              \
            PrintRecentAclError();                                                               \
            rc = 1;                                                                              \
            goto cleanup;                                                                        \
        }                                                                                        \
    } while (0)

#define RT_CHECK(expr)                                                                           \
    do {                                                                                         \
        const rtError_t _ret = (expr);                                                           \
        if (_ret != RT_ERROR_NONE) {                                                             \
            std::fprintf(stderr, "[ERROR] %s failed: %d (%s:%d)\n", #expr, (int)_ret, __FILE__, \
                         __LINE__);                                                              \
            rc = 2;                                                                              \
            goto cleanup;                                                                        \
        }                                                                                        \
    } while (0)

template <typename T>
void InitInputs(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c)
{
    for (size_t i = 0; i < a.size(); ++i) {
        a[i] = static_cast<T>(-777);
        b[i] = static_cast<T>((static_cast<int>(i % 97) - 31) * 0.01);
        c[i] = static_cast<T>((static_cast<int>(i % 53) + 3) * 0.02);
    }
}

template <>
void InitInputs<int32_t>(std::vector<int32_t>& a, std::vector<int32_t>& b, std::vector<int32_t>& c)
{
    for (size_t i = 0; i < a.size(); ++i) {
        a[i] = -777;
        b[i] = static_cast<int32_t>(i);
        c[i] = static_cast<int32_t>(1000 + i);
    }
}

template <typename T>
void BuildGolden(const std::vector<T>& b, const std::vector<T>& c, std::vector<T>& golden,
                 const std::string& mode)
{
    if (mode == "add") {
        for (size_t i = 0; i < golden.size(); ++i) {
            golden[i] = static_cast<T>(b[i] + c[i]);
        }
    } else if (mode == "exp_mul") {
        for (size_t i = 0; i < golden.size(); ++i) {
            golden[i] = static_cast<T>(std::exp(static_cast<double>(b[i])) * static_cast<double>(c[i]));
        }
    }
}

template <typename T>
int CheckOutput(const std::vector<T>& actual, const std::vector<T>& golden, const std::string& mode,
                double atol, double rtol)
{
    if (mode == "none") {
        std::printf("[CHECK] skipped\n");
        return 0;
    }

    size_t mismatch = 0;
    size_t firstMismatch = 0;
    double maxAbs = 0.0;
    double maxRel = 0.0;
    size_t maxIndex = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        const double a = static_cast<double>(actual[i]);
        const double g = static_cast<double>(golden[i]);
        const double absErr = std::fabs(a - g);
        const double relErr = absErr / std::max(std::fabs(g), atol);
        if (absErr > maxAbs) {
            maxAbs = absErr;
            maxRel = relErr;
            maxIndex = i;
        }
        if (absErr > (atol + rtol * std::fabs(g))) {
            if (mismatch == 0) {
                firstMismatch = i;
            }
            ++mismatch;
        }
    }

    std::printf("[CHECK] mode=%s maxAbs=%.10g maxRel=%.10g maxIndex=%zu mismatch=%zu/%zu\n",
                mode.c_str(), maxAbs, maxRel, maxIndex, mismatch, actual.size());
    if (mismatch != 0) {
        std::printf("[CHECK] first_mismatch idx=%zu actual=%.10g expected=%.10g\n",
                    firstMismatch, static_cast<double>(actual[firstMismatch]),
                    static_cast<double>(golden[firstMismatch]));
        return 3;
    }
    std::printf("[CHECK] PASS\n");
    return 0;
}

template <typename T>
int RunTyped(const std::string& kernelBinPath, const std::string& kernelName, size_t elems,
             int blockDim, uint32_t localMemorySize, const std::string& goldenMode, double atol, double rtol)
{
    const size_t bytes = elems * sizeof(T);
    int rc = 0;
    int deviceId = 0;
    bool aclInited = false;
    bool deviceSet = false;
    rtStream_t stream = nullptr;
    void* binHandle = nullptr;
    void* stubFunc = nullptr;
    void* dataADev = nullptr;
    void* dataBDev = nullptr;
    void* dataCDev = nullptr;
    rtDevBinary_t binary {};
    std::vector<unsigned char> kernelBuffer;
    std::vector<T> dataA(elems);
    std::vector<T> dataB(elems);
    std::vector<T> dataC(elems);
    std::vector<T> golden(elems);

    InitInputs(dataA, dataB, dataC);
    BuildGolden(dataB, dataC, golden, goldenMode);

    if (const char* envDevice = std::getenv("ACL_DEVICE_ID")) {
        deviceId = std::atoi(envDevice);
    }

    if (!ReadBinaryFile(kernelBinPath, kernelBuffer)) {
        return 66;
    }

    ACL_CHECK(aclInit(nullptr));
    aclInited = true;
    RT_CHECK(rtSetDevice(deviceId));
    deviceSet = true;
    RT_CHECK(rtStreamCreate(&stream, 0));

    RT_CHECK(rtMalloc(&dataADev, bytes, RT_MEMORY_HBM, 0));
    RT_CHECK(rtMalloc(&dataBDev, bytes, RT_MEMORY_HBM, 0));
    RT_CHECK(rtMalloc(&dataCDev, bytes, RT_MEMORY_HBM, 0));
    RT_CHECK(rtMemcpy(dataADev, bytes, dataA.data(), bytes, RT_MEMCPY_HOST_TO_DEVICE));
    RT_CHECK(rtMemcpy(dataBDev, bytes, dataB.data(), bytes, RT_MEMCPY_HOST_TO_DEVICE));
    RT_CHECK(rtMemcpy(dataCDev, bytes, dataC.data(), bytes, RT_MEMCPY_HOST_TO_DEVICE));

    binary.magic = RT_DEV_BINARY_MAGIC_ELF_AIVEC;
    binary.version = 0;
    binary.length = static_cast<uint64_t>(kernelBuffer.size());
    binary.data = kernelBuffer.data();
    RT_CHECK(rtDevBinaryRegister(&binary, &binHandle));
    RT_CHECK(rtFunctionRegister(binHandle,
                                reinterpret_cast<const void*>(kernelName.c_str()),
                                reinterpret_cast<const char_t*>(kernelName.c_str()),
                                reinterpret_cast<const void*>(kernelName.c_str()),
                                0));
    RT_CHECK(rtGetFunctionByName(reinterpret_cast<const char_t*>(kernelName.c_str()), &stubFunc));

    {
        void* args[] = {dataADev, dataBDev, dataCDev};
        if (localMemorySize == 0) {
            RT_CHECK(rtKernelLaunch(stubFunc, static_cast<uint32_t>(blockDim), args, sizeof(args), nullptr, stream));
        } else {
            aclrtLaunchKernelAttr attr {};
            // The LOCAL_MEMORY_SIZE aliases work in both CANN 9.0 beta.1 and 9.0.0.
            attr.id = ACL_RT_LAUNCH_KERNEL_ATTR_LOCAL_MEMORY_SIZE;
            attr.value.localMemorySize = localMemorySize;
            aclrtLaunchKernelCfg cfg {&attr, 1};
            ACL_CHECK(aclrtLaunchKernelV2(static_cast<aclrtFuncHandle>(stubFunc), static_cast<uint32_t>(blockDim),
                                          args, sizeof(args), &cfg, static_cast<aclrtStream>(stream)));
        }
    }
    RT_CHECK(rtStreamSynchronize(stream));
    RT_CHECK(rtMemcpy(dataA.data(), bytes, dataADev, bytes, RT_MEMCPY_DEVICE_TO_HOST));

    std::printf("[INFO] kernel=%s elems=%zu block_dim=%d local_memory_size=%u bytes=%zu\n",
                kernelName.c_str(), elems, blockDim, localMemorySize, bytes);
    for (size_t i = 0; i < std::min<size_t>(8, elems); ++i) {
        std::printf("  out[%zu]=%.10g", i, static_cast<double>(dataA[i]));
        if (goldenMode != "none") {
            std::printf(" golden=%.10g", static_cast<double>(golden[i]));
        }
        std::printf("\n");
    }

    rc = CheckOutput(dataA, golden, goldenMode, atol, rtol);

    (void)WriteFile("output0.bin", dataA.data(), bytes);
    (void)WriteFile("input_b.bin", dataB.data(), bytes);
    (void)WriteFile("input_c.bin", dataC.data(), bytes);
    if (goldenMode != "none") {
        (void)WriteFile("golden.bin", golden.data(), bytes);
    }

cleanup:
    if (dataCDev != nullptr) {
        (void)rtFree(dataCDev);
    }
    if (dataBDev != nullptr) {
        (void)rtFree(dataBDev);
    }
    if (dataADev != nullptr) {
        (void)rtFree(dataADev);
    }
    if (stream != nullptr) {
        (void)rtStreamDestroy(stream);
    }
    if (deviceSet) {
        (void)rtDeviceReset(deviceId);
    }
    if (aclInited) {
        (void)aclFinalize();
    }
    return rc;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 8) {
        std::fprintf(stderr,
                     "Usage: %s <kernel-bin> <kernel-name> <dtype:int32|fp32> <elems> <block-dim> <local-memory-size> "
                     "<golden:none|add|exp_mul> [atol] [rtol]\n",
                     argv[0]);
        return 64;
    }

    const std::string kernelBinPath = argv[1];
    const std::string kernelName = argv[2];
    const std::string dtype = argv[3];
    const size_t elems = static_cast<size_t>(std::strtoull(argv[4], nullptr, 10));
    const int blockDim = std::atoi(argv[5]);
    const uint32_t localMemorySize = static_cast<uint32_t>(std::strtoul(argv[6], nullptr, 10));
    const std::string goldenMode = argv[7];
    const double atol = argc > 8 ? std::atof(argv[8]) : 1e-5;
    const double rtol = argc > 9 ? std::atof(argv[9]) : 1e-5;

    if (dtype == "int32") {
        return RunTyped<int32_t>(kernelBinPath, kernelName, elems, blockDim, localMemorySize, goldenMode, 0.0, 0.0);
    }
    if (dtype == "fp32" || dtype == "float32") {
        return RunTyped<float>(kernelBinPath, kernelName, elems, blockDim, localMemorySize, goldenMode, atol, rtol);
    }

    std::fprintf(stderr, "[ERROR] unsupported dtype: %s\n", dtype.c_str());
    return 65;
}
