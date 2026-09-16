//
//  HybridLiteRTLM+Streaming.swift
//  react-native-litert-lm
//
//  Streaming context and callback runners for the unified execute pipeline.
//

import Foundation
import NitroModules
import CLiteRTLM

class ExecuteStreamContext {
    let userLabel: String
    let startTime: Date
    let onToken: (_ token: String, _ done: Bool) -> Void
    let promise: Promise<ExecuteResult>
    let parent: HybridLiteRTLM
    let cleanup: () -> Void
    var rawResponse: String = ""
    var fullResponse: String = ""
    var lastEmittedLength: Int = 0
    var tokenCount: Int = 0
    /// Parsed by the engine; they arrive whole, not token by token.
    var toolCalls: [ToolCall] = []
    var thinking: String = ""

    init(
        userLabel: String,
        startTime: Date,
        onToken: @escaping (_ token: String, _ done: Bool) -> Void,
        promise: Promise<ExecuteResult>,
        parent: HybridLiteRTLM,
        cleanup: @escaping () -> Void
    ) {
        self.userLabel = userLabel
        self.startTime = startTime
        self.onToken = onToken
        self.promise = promise
        self.parent = parent
        self.cleanup = cleanup
    }
}

extension HybridLiteRTLM {

    func runExecuteStreaming(
        conversation: OpaquePointer,
        msgJson: String,
        userLabel: String,
        onToken: @escaping (_ token: String, _ done: Bool) -> Void,
        promise: Promise<ExecuteResult>,
        cleanup: @escaping () -> Void
    ) {
        let ctx = ExecuteStreamContext(
            userLabel: userLabel, startTime: Date(),
            onToken: onToken, promise: promise, parent: self,
            cleanup: cleanup
        )
        let ptr = Unmanaged.passRetained(ctx).toOpaque()

        // v0.15: the callback receives an opaque LiteRtLmStreamChunk; text,
        // finality, and errors are read through accessors. The chunk (and any
        // string it returns) is only valid for the duration of the call.
        let cb: LiteRtLmStreamCallback = { ptr, chunk in
            guard let ptr = ptr, let chunk = chunk else { return }
            let ctx = Unmanaged<ExecuteStreamContext>.fromOpaque(ptr).takeUnretainedValue()

            if let errorMsg = litert_lm_stream_chunk_get_error(chunk) {
                let msg = String(cString: errorMsg)
                // stopGeneration() ends the stream with a "Task cancelled"
                // error. That is the reader pressing stop, not a failure:
                // finish with what streamed so far, as Android does, instead
                // of writing "Error: …" into the reply.
                if msg.lowercased().contains("cancel") {
                    ctx.parent.finalizeExecuteStream(ctx: ctx, streamPtr: ptr)
                    return
                }
                ctx.onToken("Error: \(msg)", true)
                ctx.cleanup()
                ctx.promise.reject(withError: NSError(domain: "LiteRTLM", code: 500,
                    userInfo: [NSLocalizedDescriptionKey: msg]))
                Unmanaged<ExecuteStreamContext>.fromOpaque(ptr).release()
                return
            }

            if litert_lm_stream_chunk_is_final(chunk) {
                ctx.parent.finalizeExecuteStream(ctx: ctx, streamPtr: ptr)
                return
            }

            if let text = litert_lm_stream_chunk_get_text(chunk) {
                ctx.parent.emitExecuteStreamChunk(ctx: ctx, chunk: text)
            }
        }

        let status = litert_lm_conversation_send_message_stream(
            conversation, msgJson, nil, nil, cb, ptr)
        if status != 0 {
            Unmanaged<ExecuteStreamContext>.fromOpaque(ptr).release()
            cleanup()
            promise.reject(withError: NSError(domain: "LiteRTLM", code: Int(status),
                userInfo: [NSLocalizedDescriptionKey: "LiteRTLM: execute streaming failed."]))
        }
    }

    func finalizeExecuteStream(ctx: ExecuteStreamContext, streamPtr: UnsafeMutableRawPointer) {
        let cleaned = stripControlTokens(ctx.rawResponse)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        var finalText = cleaned
        if !ctx.userLabel.isEmpty && finalText.hasPrefix(ctx.userLabel) {
            finalText = String(finalText.dropFirst(ctx.userLabel.count))
                .trimmingCharacters(in: .whitespacesAndNewlines)
        }
        if finalText.count > ctx.lastEmittedLength {
            let si = finalText.index(finalText.startIndex, offsetBy: ctx.lastEmittedLength)
            ctx.onToken(String(finalText[si...]), false)
        }
        ctx.fullResponse = finalText

        queue.async {
            self.commitExecuteTurn(
                userLabel: ctx.userLabel,
                modelResponse: ctx.fullResponse,
                startTime: ctx.startTime,
                conversation: self.conversation,
                tokenCount: ctx.tokenCount
            )
            ctx.onToken("", true)
            ctx.cleanup()
            ctx.promise.resolve(withResult: ExecuteResult(
                text: ctx.fullResponse, toolCalls: ctx.toolCalls, thinkingText: ctx.thinking))
            Unmanaged<ExecuteStreamContext>.fromOpaque(streamPtr).release()
        }
    }

    func emitExecuteStreamChunk(ctx: ExecuteStreamContext, chunk: UnsafePointer<CChar>) {
        let message = parseEngineMessage(String(cString: chunk))
        ctx.toolCalls.append(contentsOf: message.toolCalls)
        if !message.thinking.isEmpty {
            ctx.thinking += message.thinking
            // An empty token while the model reasons, as Android sends one:
            // the app reads it as "Thinking…" rather than a stalled reply.
            ctx.onToken("", false)
        }
        // A tool call or thinking chunk has no reply text to show.
        guard !message.text.isEmpty else { return }
        ctx.rawResponse += message.text
        let cleaned = stripControlTokens(ctx.rawResponse)
            .trimmingLeadingCharacters(in: .whitespacesAndNewlines)
        var processed = cleaned
        if !ctx.userLabel.isEmpty && processed.hasPrefix(ctx.userLabel) {
            processed = String(processed.dropFirst(ctx.userLabel.count))
                .trimmingLeadingCharacters(in: .whitespacesAndNewlines)
        }
        let safe = safeEmitLength(processed)
        if safe > ctx.lastEmittedLength {
            let chars = Array(processed)
            ctx.onToken(String(chars[ctx.lastEmittedLength..<safe]), false)
            ctx.lastEmittedLength = safe
            ctx.tokenCount += 1
        }
    }
}

private extension String {
    func trimmingLeadingCharacters(in characterSet: CharacterSet) -> String {
        guard let index = firstIndex(where: { char in
            !char.unicodeScalars.allSatisfy { characterSet.contains($0) }
        }) else {
            return ""
        }
        return String(self[index...])
    }
}
