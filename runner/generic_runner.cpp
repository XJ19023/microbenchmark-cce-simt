#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "runtime/kernel.h"
#include "runtime/rt.h"

namespace {

struct Tensor {
    std::string direction;
    std::string name;
    std::string dtype;
    size_t elements = 0;
    size_t elementSize = 0;
    std::vector<unsigned char> host;
    void* device = nullptr;
};

bool IsInput(const Tensor& tensor)
{
    return tensor.direction == "input" || tensor.direction == "inout";
}

bool IsOutput(const Tensor& tensor)
{
    return tensor.direction == "output" || tensor.direction == "inout";
}

size_t DtypeSize(const std::string& dtype)
{
    if (dtype == "int8" || dtype == "uint8") return 1;
    if (dtype == "int16" || dtype == "uint16" || dtype == "fp16" || dtype == "bf16") return 2;
    if (dtype == "int32" || dtype == "uint32" || dtype == "fp32") return 4;
    if (dtype == "int64" || dtype == "uint64" || dtype == "fp64") return 8;
    return 0;
}

bool ReadBinaryFile(const std::string& path, std::vector<unsigned char>& buffer)
{
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) return false;
    file.seekg(0, std::ios::end);
    const std::streamoff size = file.tellg();
    file.seekg(0, std::ios::beg);
    if (size <= 0) return false;
    buffer.resize(static_cast<size_t>(size));
    file.read(reinterpret_cast<char*>(buffer.data()), size);
    return static_cast<bool>(file);
}

bool WriteFile(const std::string& path, const void* data, size_t bytes)
{
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) return false;
    file.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    return static_cast<bool>(file);
}

bool ReadTensorSpec(const std::string& path, std::vector<Tensor>& tensors)
{
    std::ifstream file(path);
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        Tensor tensor;
        std::istringstream fields(line);
        if (!(fields >> tensor.direction >> tensor.name >> tensor.dtype >> tensor.elements)) {
            std::fprintf(stderr, "[ERROR] invalid tensor spec line: %s\n", line.c_str());
            return false;
        }
        tensor.elementSize = DtypeSize(tensor.dtype);
        if ((tensor.direction != "input" && tensor.direction != "output" && tensor.direction != "inout") ||
            tensor.elementSize == 0 || tensor.elements == 0) {
            std::fprintf(stderr, "[ERROR] invalid tensor spec values: %s\n", line.c_str());
            return false;
        }
        tensor.host.resize(tensor.elements * tensor.elementSize);
        tensors.push_back(std::move(tensor));
    }
    return !tensors.empty();
}

uint16_t FloatToHalf(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000;
    int exponent = static_cast<int>((bits >> 23) & 0xff) - 127 + 15;
    uint32_t mantissa = bits & 0x7fffff;
    if (exponent <= 0) return static_cast<uint16_t>(sign);
    if (exponent >= 31) return static_cast<uint16_t>(sign | 0x7c00);
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) | (mantissa >> 13));
}

template <typename T>
void Fill(std::vector<unsigned char>& data, size_t elements, size_t tensorIndex)
{
    T* values = reinterpret_cast<T*>(data.data());
    for (size_t i = 0; i < elements; ++i) {
        values[i] = static_cast<T>((i % 97) + 1 + tensorIndex * 101);
    }
}

void InitTensor(Tensor& tensor, size_t tensorIndex)
{
    if (!IsInput(tensor)) {
        std::memset(tensor.host.data(), 0xcd, tensor.host.size());
        return;
    }
    if (tensor.dtype == "int8") Fill<int8_t>(tensor.host, tensor.elements, tensorIndex);
    else if (tensor.dtype == "uint8") Fill<uint8_t>(tensor.host, tensor.elements, tensorIndex);
    else if (tensor.dtype == "int16") Fill<int16_t>(tensor.host, tensor.elements, tensorIndex);
    else if (tensor.dtype == "uint16") Fill<uint16_t>(tensor.host, tensor.elements, tensorIndex);
    else if (tensor.dtype == "int32") Fill<int32_t>(tensor.host, tensor.elements, tensorIndex);
    else if (tensor.dtype == "uint32") Fill<uint32_t>(tensor.host, tensor.elements, tensorIndex);
    else if (tensor.dtype == "int64") Fill<int64_t>(tensor.host, tensor.elements, tensorIndex);
    else if (tensor.dtype == "uint64") Fill<uint64_t>(tensor.host, tensor.elements, tensorIndex);
    else if (tensor.dtype == "fp32") {
        float* values = reinterpret_cast<float*>(tensor.host.data());
        for (size_t i = 0; i < tensor.elements; ++i) values[i] = (static_cast<int>(i % 97) - 31) * 0.01f + tensorIndex;
    } else if (tensor.dtype == "fp64") {
        double* values = reinterpret_cast<double*>(tensor.host.data());
        for (size_t i = 0; i < tensor.elements; ++i) values[i] = (static_cast<int>(i % 97) - 31) * 0.01 + tensorIndex;
    } else if (tensor.dtype == "fp16" || tensor.dtype == "bf16") {
        uint16_t* values = reinterpret_cast<uint16_t*>(tensor.host.data());
        for (size_t i = 0; i < tensor.elements; ++i) {
            const float value = (static_cast<int>(i % 97) - 31) * 0.01f + tensorIndex;
            if (tensor.dtype == "fp16") {
                values[i] = FloatToHalf(value);
            } else {
                uint32_t bits = 0;
                std::memcpy(&bits, &value, sizeof(bits));
                values[i] = static_cast<uint16_t>(bits >> 16);
            }
        }
    }
}

double ValueAt(const Tensor& tensor, size_t index)
{
    const unsigned char* data = tensor.host.data();
    if (tensor.dtype == "int8") return reinterpret_cast<const int8_t*>(data)[index];
    if (tensor.dtype == "uint8") return reinterpret_cast<const uint8_t*>(data)[index];
    if (tensor.dtype == "int16") return reinterpret_cast<const int16_t*>(data)[index];
    if (tensor.dtype == "uint16") return reinterpret_cast<const uint16_t*>(data)[index];
    if (tensor.dtype == "int32") return reinterpret_cast<const int32_t*>(data)[index];
    if (tensor.dtype == "uint32") return reinterpret_cast<const uint32_t*>(data)[index];
    if (tensor.dtype == "int64") return static_cast<double>(reinterpret_cast<const int64_t*>(data)[index]);
    if (tensor.dtype == "uint64") return static_cast<double>(reinterpret_cast<const uint64_t*>(data)[index]);
    if (tensor.dtype == "fp32") return reinterpret_cast<const float*>(data)[index];
    if (tensor.dtype == "fp64") return reinterpret_cast<const double*>(data)[index];
    return reinterpret_cast<const uint16_t*>(data)[index];
}

void PrintRecentAclError()
{
    const char* recent = aclGetRecentErrMsg();
    if (recent != nullptr && recent[0] != '\0') std::fprintf(stderr, "[ACL] RecentErrMsg: %s\n", recent);
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

int Run(const std::string& kernelBinPath, const std::string& kernelName, const std::string& specPath,
        uint32_t blockDim, uint32_t localMemorySize)
{
    int rc = 0;
    int deviceId = 0;
    bool aclInited = false;
    bool deviceSet = false;
    rtStream_t stream = nullptr;
    void* binHandle = nullptr;
    void* stubFunc = nullptr;
    rtDevBinary_t binary {};
    std::vector<unsigned char> kernelBuffer;
    std::vector<Tensor> tensors;
    std::vector<void*> launchArgs;

    if (!ReadTensorSpec(specPath, tensors)) {
        std::fprintf(stderr, "[ERROR] failed to read tensor spec: %s\n", specPath.c_str());
        return 65;
    }
    if (!ReadBinaryFile(kernelBinPath, kernelBuffer)) {
        std::fprintf(stderr, "[ERROR] failed to read kernel binary: %s\n", kernelBinPath.c_str());
        return 66;
    }
    for (size_t i = 0; i < tensors.size(); ++i) {
        InitTensor(tensors[i], i);
        if (IsInput(tensors[i])) {
            (void)WriteFile("input_" + tensors[i].name + ".bin", tensors[i].host.data(), tensors[i].host.size());
        }
    }
    launchArgs.resize(tensors.size(), nullptr);
    if (const char* envDevice = std::getenv("ACL_DEVICE_ID")) deviceId = std::atoi(envDevice);

    ACL_CHECK(aclInit(nullptr));
    aclInited = true;
    RT_CHECK(rtSetDevice(deviceId));
    deviceSet = true;
    RT_CHECK(rtStreamCreate(&stream, 0));
    for (size_t i = 0; i < tensors.size(); ++i) {
        RT_CHECK(rtMalloc(&tensors[i].device, tensors[i].host.size(), RT_MEMORY_HBM, 0));
        RT_CHECK(rtMemcpy(tensors[i].device, tensors[i].host.size(), tensors[i].host.data(),
                          tensors[i].host.size(), RT_MEMCPY_HOST_TO_DEVICE));
        launchArgs[i] = tensors[i].device;
    }

    binary.magic = RT_DEV_BINARY_MAGIC_ELF_AIVEC;
    binary.version = 0;
    binary.length = static_cast<uint64_t>(kernelBuffer.size());
    binary.data = kernelBuffer.data();
    RT_CHECK(rtDevBinaryRegister(&binary, &binHandle));
    RT_CHECK(rtFunctionRegister(binHandle, reinterpret_cast<const void*>(kernelName.c_str()),
                                reinterpret_cast<const char_t*>(kernelName.c_str()),
                                reinterpret_cast<const void*>(kernelName.c_str()), 0));
    RT_CHECK(rtGetFunctionByName(reinterpret_cast<const char_t*>(kernelName.c_str()), &stubFunc));

    if (localMemorySize == 0) {
        RT_CHECK(rtKernelLaunch(stubFunc, blockDim, launchArgs.data(),
                                launchArgs.size() * sizeof(void*), nullptr, stream));
    } else {
        aclrtLaunchKernelAttr attr {};
        attr.id = ACL_RT_LAUNCH_KERNEL_ATTR_LOCAL_MEMORY_SIZE;
        attr.value.localMemorySize = localMemorySize;
        aclrtLaunchKernelCfg cfg {&attr, 1};
        ACL_CHECK(aclrtLaunchKernelV2(static_cast<aclrtFuncHandle>(stubFunc), blockDim, launchArgs.data(),
                                      launchArgs.size() * sizeof(void*), &cfg,
                                      static_cast<aclrtStream>(stream)));
    }
    RT_CHECK(rtStreamSynchronize(stream));

    std::printf("[INFO] kernel=%s block_dim=%u local_memory_size=%u tensor_args=%zu\n",
                kernelName.c_str(), blockDim, localMemorySize, tensors.size());
    for (Tensor& tensor : tensors) {
        if (!IsOutput(tensor)) continue;
        RT_CHECK(rtMemcpy(tensor.host.data(), tensor.host.size(), tensor.device, tensor.host.size(),
                          RT_MEMCPY_DEVICE_TO_HOST));
        const std::string outputPath = "output_" + tensor.name + ".bin";
        (void)WriteFile(outputPath, tensor.host.data(), tensor.host.size());
        std::printf("[OUTPUT] %s dtype=%s elements=%zu file=%s\n", tensor.name.c_str(),
                    tensor.dtype.c_str(), tensor.elements, outputPath.c_str());
        for (size_t i = 0; i < std::min<size_t>(8, tensor.elements); ++i) {
            std::printf("  %s[%zu]=%.10g\n", tensor.name.c_str(), i, ValueAt(tensor, i));
        }
    }
    std::printf("[CHECK] skipped (generic tensor mode)\n");

cleanup:
    for (Tensor& tensor : tensors) {
        if (tensor.device != nullptr) (void)rtFree(tensor.device);
    }
    if (stream != nullptr) (void)rtStreamDestroy(stream);
    if (deviceSet) (void)rtDeviceReset(deviceId);
    if (aclInited) (void)aclFinalize();
    return rc;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc != 6) {
        std::fprintf(stderr,
                     "Usage: %s <kernel-bin> <kernel-name> <tensor-spec.tsv> <block-dim> <local-memory-size>\n",
                     argv[0]);
        return 64;
    }
    const uint32_t blockDim = static_cast<uint32_t>(std::strtoul(argv[4], nullptr, 10));
    const uint32_t localMemorySize = static_cast<uint32_t>(std::strtoul(argv[5], nullptr, 10));
    return Run(argv[1], argv[2], argv[3], blockDim, localMemorySize);
}
