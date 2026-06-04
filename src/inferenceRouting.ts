import type { MultimodalPart, PartType } from "./specs/LiteRTLM.nitro";

export type TokenCallback = (token: string, done: boolean) => void;

/** Legacy inference entry points on LiteRTLM (createLLM routes these to execute). */
export const LEGACY_INFERENCE_METHODS = [
  "sendMessage",
  "sendMessageWithImage",
  "sendMessageWithAudio",
  "sendMultimodalMessage",
  "sendMessageAsync",
  "sendMessageWithImageAsync",
  "sendMessageWithAudioAsync",
] as const;

export type LegacyInferenceMethod = (typeof LEGACY_INFERENCE_METHODS)[number];

export function textPart(text: string): MultimodalPart {
  return { type: "text" as PartType, text };
}

export function imagePart(path: string): MultimodalPart {
  return { type: "image" as PartType, path };
}

export function audioPart(path: string): MultimodalPart {
  return { type: "audio" as PartType, path };
}

export type InferenceRoute = {
  parts: MultimodalPart[];
  onToken?: TokenCallback;
};

type LegacyRouteBuilder = (args: unknown[]) => InferenceRoute;

const LEGACY_ROUTES: Record<LegacyInferenceMethod, LegacyRouteBuilder> = {
  sendMessage: (args) => ({ parts: [textPart(args[0] as string)] }),
  sendMessageWithImage: (args) => ({
    parts: [textPart(args[0] as string), imagePart(args[1] as string)],
  }),
  sendMessageWithAudio: (args) => ({
    parts: [textPart(args[0] as string), audioPart(args[1] as string)],
  }),
  sendMultimodalMessage: (args) => ({ parts: args[0] as MultimodalPart[] }),
  sendMessageAsync: (args) => ({
    parts: [textPart(args[0] as string)],
    onToken: args[1] as TokenCallback,
  }),
  sendMessageWithImageAsync: (args) => ({
    parts: [textPart(args[0] as string), imagePart(args[1] as string)],
    onToken: args[2] as TokenCallback,
  }),
  sendMessageWithAudioAsync: (args) => ({
    parts: [textPart(args[0] as string), audioPart(args[1] as string)],
    onToken: args[2] as TokenCallback,
  }),
};

const LEGACY_METHOD_SET = new Set<string>(LEGACY_INFERENCE_METHODS);

/**
 * Maps legacy public API names to unified MultimodalPart payloads.
 * JS source of truth — native legacy wrappers mirror these shapes for direct hybrid access.
 */
export function routeLegacyInference(
  method: string,
  args: unknown[],
): InferenceRoute | null {
  if (!LEGACY_METHOD_SET.has(method)) {
    return null;
  }
  return LEGACY_ROUTES[method as LegacyInferenceMethod](args);
}

export function isLegacyInferenceMethod(
  method: string,
): method is LegacyInferenceMethod {
  return LEGACY_METHOD_SET.has(method);
}
