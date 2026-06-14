import { NitroModules } from "react-native-nitro-modules";
import { LiteRTLM, LLMConfig, MultimodalPart } from "./specs/LiteRTLM.nitro";
import { createMemoryTracker, MemoryTracker } from "./memoryTracker";
import { ModelRegistry } from "./modelRegistry";
import {
  isLegacyInferenceMethod,
  routeLegacyInference,
  TokenCallback,
} from "./inferenceRouting";

/**
 * Extended LiteRT-LM instance with optional memory tracking and
 * augmented loadModel that accepts a download progress callback.
 */
export type LiteRTLMInstance = Omit<LiteRTLM, "loadModel"> & {
  memoryTracker?: MemoryTracker;
  loadModel: (
    pathOrUrl: string,
    config?: LLMConfig,
    onDownloadProgress?: (progress: number) => void,
  ) => Promise<void>;
};

/**
 * Creates a new LiteRT-LM inference engine instance.
 *
 * Optionally creates a native-backed memory tracker using
 * `NitroModules.createNativeArrayBuffer()` (v0.35+) for efficient
 * zero-copy memory usage tracking.
 *
 * @param options.enableMemoryTracking Enable automatic memory tracking (default: false)
 * @param options.maxMemorySnapshots Maximum number of memory snapshots to store (default: 256)
 */
export function createLLM(options?: {
  enableMemoryTracking?: boolean;
  maxMemorySnapshots?: number;
}): LiteRTLMInstance {
  const native = NitroModules.createHybridObject<LiteRTLM>("LiteRTLM");

  const enableTracking = options?.enableMemoryTracking ?? false;
  const tracker = enableTracking
    ? createMemoryTracker(options?.maxMemorySnapshots ?? 256)
    : undefined;

  const recordMemorySnapshot = () => {
    if (!tracker) return;
    try {
      const usage = native.getMemoryUsage();
      tracker.record({
        timestamp: Date.now(),
        nativeHeapBytes: usage.nativeHeapBytes,
        residentBytes: usage.residentBytes,
        availableMemoryBytes: usage.availableMemoryBytes,
      });
    } catch {
      // Non-critical
    }
  };

  const augmentedLoadModel = async (
    pathOrUrl: string,
    config?: LLMConfig,
    onDownloadProgress?: (progress: number) => void,
  ) => {
    console.log(`Resolving model at ${pathOrUrl}...`);
    const modelPath = await ModelRegistry.resolveModel(pathOrUrl, {
      onProgress: onDownloadProgress,
    });
    console.log(`Model resolved to: ${modelPath}`);

    const result = await native.loadModel(modelPath, config);

    if (tracker) {
      tracker.reset();
      recordMemorySnapshot();
    }

    return result;
  };

  /** Single JS inference path — always calls native execute(). */
  const runExecute = (
    parts: MultimodalPart[],
    onToken?: TokenCallback,
  ): Promise<string> => {
    const processedParts = parts.map((part) => {
      if (part.path?.startsWith("file://")) {
        return { ...part, path: part.path.substring(7) };
      }
      return part;
    });

    if (onToken) {
      const wrapped: TokenCallback = (token, done) => {
        onToken(token, done);
        if (done) recordMemorySnapshot();
      };
      return native.execute(processedParts, wrapped);
    }
    return native.execute(processedParts, undefined).then((result: string) => {
      recordMemorySnapshot();
      return result;
    });
  };

  return new Proxy(native, {
    get(target, prop, receiver) {
      if (typeof prop !== "string") {
        return Reflect.get(target, prop, receiver);
      }

      if (prop === "memoryTracker") {
        return tracker;
      }
      if (prop === "loadModel") {
        return augmentedLoadModel;
      }
      if (prop === "execute") {
        return (parts: MultimodalPart[], onToken?: TokenCallback) =>
          runExecute(parts, onToken);
      }
      if (prop === "resetConversation") {
        return () => {
          const result = target.resetConversation();
          recordMemorySnapshot();
          return result;
        };
      }

      if (isLegacyInferenceMethod(prop)) {
        return (...args: unknown[]) => {
          const route = routeLegacyInference(prop, args)!;
          const promise = runExecute(route.parts, route.onToken);
          return prop.endsWith("Async") ? promise.then(() => {}) : promise;
        };
      }

      const original = target[prop as keyof LiteRTLM];
      if (typeof original === "function") {
        return original.bind(target);
      }
      return original;
    },
  }) as unknown as LiteRTLMInstance;
}
