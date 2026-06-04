import {
  routeLegacyInference,
  isLegacyInferenceMethod,
  textPart,
  imagePart,
  audioPart,
} from "../inferenceRouting";

describe("inferenceRouting", () => {
  it("routes sendMessage to a single text part", () => {
    const route = routeLegacyInference("sendMessage", ["hello"]);
    expect(route).toEqual({ parts: [textPart("hello")] });
  });

  it("routes sendMessageWithImageAsync with stream callback", () => {
    const onToken = jest.fn();
    const route = routeLegacyInference("sendMessageWithImageAsync", [
      "describe",
      "/img.jpg",
      onToken,
    ]);
    expect(route).toEqual({
      parts: [textPart("describe"), imagePart("/img.jpg")],
      onToken,
    });
  });

  it("returns null for unknown methods", () => {
    expect(routeLegacyInference("downloadModel", ["url", "file"])).toBeNull();
  });

  it("isLegacyInferenceMethod narrows known methods", () => {
    expect(isLegacyInferenceMethod("sendMessage")).toBe(true);
    expect(isLegacyInferenceMethod("close")).toBe(false);
  });
});
