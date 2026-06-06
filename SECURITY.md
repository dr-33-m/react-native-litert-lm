# Security Policy

This document outlines the security practices, vulnerability reporting process, and local execution guarantees for `react-native-litert-lm`.

## Supported Versions

We actively support and patch the current minor release branch.

| Version  | Supported |
| -------- | --------- |
| >= 0.4.x | ✅ Yes    |
| < 0.4.x  | ❌ No     |

Please upgrade to the latest stable release to ensure you have all security patches, path traversal fixes, and performance updates.

---

## On-Device Data Privacy (Local-First by Design)

`react-native-litert-lm` is built for **100% on-device local execution**:

- **Zero Remote Telemetry**: Prompt inputs, model generated outputs, images, and audio buffers never leave the host device.
- **No Third-Party APIs**: All tensor operations and natural language processing occur locally in the compiled LiteRT-LM C++ runtime (`CLiteRTLM.xcframework` on iOS and Android Maven dependencies).
- **Compliance**: This architecture inherently simplifies compliance with strict data-privacy standards such as GDPR, HIPAA, and CCPA.

---

## Key Security Controls & Defenses

### 1. Transport Security (HTTPS Enforcement)

To defend against Man-in-the-Middle (MitM) attacks during model weight acquisition, the library strictly enforces HTTPS:

- Non-HTTPS (`http://`) URL schemes are rejected at both the TypeScript factory level (`createLLM`) and inside the native iOS (`URLSession`) and Android download handlers.
- If an insecure model download URL is provided, the library will throw an exception and block the request.

### 2. Path Traversal & Arbitrary File Deletion Protections

To prevent malicious applications or inputs from reading or deleting files outside the sandbox:

- Any filename parameter passed to `downloadModel` or `deleteModel` is sanitized.
- Filenames containing path traversal segments (e.g., `..`, `/`, `\`) are strictly blocked, raising an `IllegalArgumentException` on Android and rejecting the Promise with a `LiteRTLM` domain error on iOS.
- Cached models are stored in app-private, platform-specific sandboxed directories:
  - **Android**: `files/models/`
  - **iOS**: `Library/Caches/litert_models/`

### 3. Model Weight Integrity & Verification

AI models (such as Gemma 4) are compiled binaries (`.litertlm`). Loaded model files could potentially exploit bugs in the underlying C++ runtime if they are malformed or maliciously modified.

- **Trusted Sources Only**: Never load `.litertlm` models from untrusted or unverified sources.
- **Manual Download Validation**: When downloading models outside the library's automated utility, we highly recommend fetching the model over HTTPS, verifying its integrity via SHA-256 checksums, and verifying authorization credentials before loading the file path into the engine.

### 4. Memory-Leak & Denials of Service (DoS) Protections

Improper native memory management in heavy on-device ML contexts can lead to Out-Of-Memory (OOM) crashes:

- **Ref-Counting & Deallocations**: Heavy graphics and media inputs (multimodal frames) and FFI pointers are manually managed and explicitly freed upon conversation reset or model release to prevent memory starvation.
- **Zero-Copy Pipelines**: Multimodal inference leverages Nitro Modules' native-owned `ArrayBuffer` directly mapping memory to avoid JavaScript heap duplicates or heavy base64 serialization.
- **Background Execution**: In order to prevent locking the React Native JSI thread, native FFI tasks are dispatched to dedicated background serial queues (`dev.litert.engine`), ensuring the application UI remains responsive.

---

## Reporting a Vulnerability

If you discover a security vulnerability in this project, please **do not open a public GitHub issue**. Instead, report it responsibly via one of the following methods:

1. **GitHub Security Advisory**: Submit a private vulnerability report through the **Security** tab of the GitHub repository.
2. **Direct Contact**: Contact the maintainer at GitHub Profile: [hung-yueh](https://github.com/hung-yueh).

Please include a detailed description of the issue, step-by-step instructions to reproduce it, and any proof-of-concept (PoC) code if available.

### Response Timeline

- **Acknowledgement**: Within 72 hours of receipt.
- **Status Updates**: Periodic updates during validation and patching.
- **Disclosure**: Public disclosure will be coordinated after a fix is prepared and released, generally within 30-90 days.
