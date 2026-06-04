//
//  HybridLiteRTLM+Execute.swift
//  react-native-litert-lm
//
//  Unified inference pipeline (execute). JS routes via src/inferenceRouting.ts;
//  legacy send* methods below delegate here for direct native hybrid access.
//

import Foundation
import UIKit
import NitroModules
import CLiteRTLM



// MARK: - Preprocessed parts for thread-safety

struct PreprocessedPart {
    enum Kind {
        case text(String)
        case imagePath(String)
        case imageData(Data)
        case audioPath(String)
        case audioData(Data)
    }
    let kind: Kind
    let label: String
}

// MARK: - Execute pipeline

extension HybridLiteRTLM {

    /// Only used when legacy native *Async methods are invoked directly (not via createLLM JS proxy).
    func executeVoid(
        parts: [MultimodalPart],
        onToken: @escaping (_ token: String, _ done: Bool) -> Void
    ) throws -> Promise<Void> {
        let voidPromise = Promise<Void>()
        let resultPromise = try execute(parts: parts, onToken: onToken)
        resultPromise
            .then { _ in voidPromise.resolve() }
            .catch { voidPromise.reject(withError: $0) }
        return voidPromise
    }

    public func execute(parts: [MultimodalPart], onToken: ((_ token: String, _ done: Bool) -> Void)?) throws -> Promise<String> {
        let promise = Promise<String>()

        // Preprocess all JSI-bound data on the caller thread synchronously
        var preprocessed: [PreprocessedPart] = []
        for part in parts {
            switch part.type {
            case .text:
                let txt = part.text ?? ""
                preprocessed.append(PreprocessedPart(kind: .text(txt), label: txt))
            case .image:
                if let path = part.path {
                    preprocessed.append(PreprocessedPart(kind: .imagePath(path), label: "[Image]"))
                } else if let buf = part.imageBuffer {
                    let data = buf.toData(copyIfNeeded: true)
                    preprocessed.append(PreprocessedPart(kind: .imageData(data), label: "[Image]"))
                }
            case .audio:
                if let path = part.path {
                    preprocessed.append(PreprocessedPart(kind: .audioPath(path), label: "[Audio]"))
                } else if let buf = part.audioBuffer {
                    let data = buf.toData(copyIfNeeded: true)
                    preprocessed.append(PreprocessedPart(kind: .audioData(data), label: "[Audio]"))
                }
            }
        }

        let userLabel = preprocessed.map { $0.label }.joined(separator: " ").trimmingCharacters(in: .whitespaces)

        queue.async {
            guard let conversation = self.conversation else {
                promise.reject(withError: NSError(domain: "LiteRTLM", code: 400,
                    userInfo: [NSLocalizedDescriptionKey: "LiteRTLM: No model loaded. Call loadModel() first."]))
                return
            }

            let payload: (json: String, tempFiles: [String])
            do { payload = try self.buildExecutePayload(preprocessed) }
            catch { promise.reject(withError: error); return }

            let msgJson = payload.json
            let tempFiles = payload.tempFiles
            let cleanup = { tempFiles.forEach { try? FileManager.default.removeItem(atPath: $0) } }

            if let onToken = onToken {
                self.runExecuteStreaming(
                    conversation: conversation,
                    msgJson: msgJson,
                    userLabel: userLabel,
                    onToken: onToken,
                    promise: promise,
                    cleanup: cleanup
                )
            } else {
                self.runExecuteBlocking(
                    conversation: conversation,
                    msgJson: msgJson,
                    userLabel: userLabel,
                    promise: promise,
                    cleanup: cleanup
                )
            }
        }

        return promise
    }

    // MARK: - Payload / media

    private func validateMediaPath(_ path: String, label: String) throws {
        if !FileManager.default.fileExists(atPath: path) {
            throw NSError(
                domain: "LiteRTLM", code: 404,
                userInfo: [NSLocalizedDescriptionKey: "\(label) file not found: \(path)"]
            )
        }
    }

    private func buildExecutePayload(_ preprocessed: [PreprocessedPart]) throws -> (json: String, tempFiles: [String]) {
        struct Desc { let kind: String; let text: String?; let file: String? }
        var descs: [Desc] = []
        var temps: [String] = []

        for part in preprocessed {
            switch part.kind {
            case .text(let text):
                descs.append(Desc(kind: "text", text: text, file: nil))
            case .imagePath(let path):
                try validateMediaPath(path, label: "Image")
                let scaled = scaleImageIfNeeded(path)
                if scaled != path { temps.append(scaled) }
                descs.append(Desc(kind: "image", text: nil, file: scaled))
            case .imageData(let data):
                let raw = try saveDataToTempFile(data, ext: "jpg")
                temps.append(raw)
                let scaled = scaleImageIfNeeded(raw)
                if scaled != raw { temps.append(scaled) }
                descs.append(Desc(kind: "image", text: nil, file: scaled))
            case .audioPath(let path):
                try validateMediaPath(path, label: "Audio")
                descs.append(Desc(kind: "audio", text: nil, file: path))
            case .audioData(let data):
                let raw = try saveDataToTempFile(data, ext: "wav")
                temps.append(raw)
                descs.append(Desc(kind: "audio", text: nil, file: raw))
            }
        }

        if descs.count == 1 && descs[0].kind == "text" {
            return (buildTextMessageJson(text: descs[0].text ?? ""), temps)
        }

        let partsJson = descs.map { d -> String in
            switch d.kind {
            case "text":  return "{\"type\":\"text\",\"text\":\"" + escapeJson(d.text ?? "") + "\"}"
            case "image": return "{\"type\":\"image\",\"path\":\"" + escapeJson(d.file ?? "") + "\"}"
            default:      return "{\"type\":\"audio\",\"path\":\"" + escapeJson(d.file ?? "") + "\"}"
            }
        }.joined(separator: ",")
        return ("{\"role\":\"user\",\"content\":[" + partsJson + "]}", temps)
    }

    private func buildTextMessageJson(text: String) -> String {
        return "{\"role\":\"user\",\"content\":\"" + escapeJson(text) + "\"}"
    }

    private func scaleImageIfNeeded(_ imagePath: String, maxDimension: Int = 1024) -> String {
        guard let image = UIImage(contentsOfFile: imagePath) else { return imagePath }
        let w = Int(image.size.width), h = Int(image.size.height)
        guard max(w, h) > maxDimension else { return imagePath }
        let scale = CGFloat(maxDimension) / CGFloat(max(w, h))
        let newSize = CGSize(width: CGFloat(w) * scale, height: CGFloat(h) * scale)
        let renderer = UIGraphicsImageRenderer(size: newSize)
        let scaled = renderer.image { _ in image.draw(in: CGRect(origin: .zero, size: newSize)) }
        let tmp = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("litert_scaled_\(Int(Date().timeIntervalSince1970 * 1_000)).jpg")
        if let data = scaled.jpegData(compressionQuality: 0.9) {
            try? data.write(to: URL(fileURLWithPath: tmp))
        }
        return tmp
    }

    private func saveDataToTempFile(_ data: Data, ext: String) throws -> String {
        let tmp = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("litert_buf_\(Int(Date().timeIntervalSince1970 * 1_000)).\(ext)")
        try data.write(to: URL(fileURLWithPath: tmp))
        return tmp
    }

    // MARK: - Streaming / blocking runners

    private func runExecuteBlocking(
        conversation: OpaquePointer,
        msgJson: String,
        userLabel: String,
        promise: Promise<String>,
        cleanup: @escaping () -> Void
    ) {
        let startTime = Date()
        guard let response = litert_lm_conversation_send_message(conversation, msgJson, nil, nil) else {
            cleanup()
            promise.reject(withError: NSError(domain: "LiteRTLM", code: 500,
                userInfo: [NSLocalizedDescriptionKey: "LiteRTLM: execute failed."]))
            return
        }
        defer { litert_lm_json_response_delete(response) }

        var result = ""
        if let rs = litert_lm_json_response_get_string(response) {
            result = extractTextFromResponse(String(cString: rs))
                .trimmingCharacters(in: .whitespacesAndNewlines)
        }

        commitExecuteTurn(
            userLabel: userLabel,
            modelResponse: result,
            startTime: startTime,
            conversation: conversation,
            tokenCount: 0
        )
        cleanup()
        promise.resolve(withResult: result)
    }



    func commitExecuteTurn(
        userLabel: String,
        modelResponse: String,
        startTime: Date,
        conversation: OpaquePointer?,
        tokenCount: Int
    ) {
        let totalTime = Date().timeIntervalSince(startTime)
        var compTok = Double(tokenCount)
        var tps = 0.0, ttft = 0.0
        if let conversation = conversation,
           let bi = litert_lm_conversation_get_benchmark_info(conversation) {
            let turns = litert_lm_benchmark_info_get_num_decode_turns(bi)
            if turns > 0 {
                let li = turns - 1
                tps = litert_lm_benchmark_info_get_decode_tokens_per_sec_at(bi, li)
                compTok = Double(litert_lm_benchmark_info_get_decode_token_count_at(bi, li))
            }
            ttft = litert_lm_benchmark_info_get_time_to_first_token(bi)
            litert_lm_benchmark_info_delete(bi)
        }
        let pt = Double(userLabel.count) / 4.0
        if compTok == 0.0 { compTok = Double(modelResponse.count) / 4.0 }
        lastStats = GenerationStats(
            promptTokens: pt, completionTokens: compTok,
            totalTokens: pt + compTok, timeToFirstToken: ttft,
            totalTime: totalTime,
            tokensPerSecond: tps > 0 ? tps : (compTok / totalTime)
        )
        history.append(Message(role: .user, content: userLabel))
        history.append(Message(role: .model, content: modelResponse))
    }
}

