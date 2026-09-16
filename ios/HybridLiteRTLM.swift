//
//  HybridLiteRTLM.swift
//  react-native-litert-lm
//
//  Created by Antigravity on 2026-05-19.
//  Copyright © 2026 Margelo. All rights reserved.
//

import Foundation
import NitroModules
import CLiteRTLM
import os

public class HybridLiteRTLM: HybridLiteRTLMSpec_base, HybridLiteRTLMSpec_protocol {
    
    // MARK: - Internal (for extension files)
    
    /// Dedicated background serial queue to protect the JSI/JS thread from blocking and deadlocks (User Rule #1).
    let queue = DispatchQueue(label: "dev.litert.engine", qos: .userInteractive)
    
    /// Opaque pointer to the active conversation state.
    var conversation: OpaquePointer?
    
    /// Conversation history.
    var history: [Message] = []
    
    /// Latest inference generation statistics.
    var lastStats = GenerationStats(
        promptTokens: 0.0,
        completionTokens: 0.0,
        totalTokens: 0.0,
        timeToFirstToken: 0.0,
        totalTime: 0.0,
        tokensPerSecond: 0.0
    )
    
    var loadedModelPath: String?
    let modelStore = HybridModelStore()
    
    // MARK: - Private state
    
    /// Opaque pointer to the LiteRT LM C Engine.
    private var engine: OpaquePointer?
    
    /// Thread-safe status flag.
    private var isLoaded = false
    
    // Default configuration variables
    private var backend: Backend = .cpu
    private var temperature: Double = 0.7
    private var topK: Int = 40
    private var topP: Double = 0.95
    private var maxContextTokens: Int = 4096
    private var maxOutputTokens: Int = 1024
    private var systemPrompt: String?
    private var tools: [ToolDefinition]?
    private var enableSpeculativeDecoding: Bool = false
    private var enableThinking: Bool = false
    
    /// Approximate model weight size to inform the JS engine's garbage collection.
    public var memorySize: Int {
        return 1024 * 1024 * 1024 // ~1GB proxy
    }
    
    deinit {
        closeInternal()
    }
    
    // MARK: - Core Hybrid Object API
    
    public func isReady() throws -> Bool {
        return queue.sync { isLoaded }
    }
    
    public func getHistory() throws -> [Message] {
        return queue.sync { history }
    }
    
    public func resetConversation() throws {
        queue.sync {
            history.removeAll()
            lastStats = GenerationStats(
                promptTokens: 0.0,
                completionTokens: 0.0,
                totalTokens: 0.0,
                timeToFirstToken: 0.0,
                totalTime: 0.0,
                tokensPerSecond: 0.0
            )
            if isLoaded && engine != nil {
                createNewConversation()
            }
        }
    }
    
    public func resetConversationWith(messages: [Message]) throws {
        queue.sync {
            history = messages
            lastStats = GenerationStats(
                promptTokens: 0.0,
                completionTokens: 0.0,
                totalTokens: 0.0,
                timeToFirstToken: 0.0,
                totalTime: 0.0,
                tokensPerSecond: 0.0
            )
            if isLoaded && engine != nil {
                // Recreating the conversation is what actually frees the old KV
                // cache; the seed is replayed into the new one without
                // triggering generation.
                createNewConversation(initialMessages: messages)
            }
        }
    }

    public func getStats() throws -> GenerationStats {
        return queue.sync { lastStats }
    }
    
    public func countTokens(text: String) throws -> Double {
        return queue.sync {
            guard let engine = self.engine else {
                return -1.0
            }
            guard let result = litert_lm_engine_tokenize(engine, text) else {
                return -1.0
            }
            let numTokens = litert_lm_tokenize_result_get_num_tokens(result)
            litert_lm_tokenize_result_delete(result)
            return Double(numTokens)
        }
    }
    
    public func getMemoryUsage() throws -> MemoryUsage {
        return Self.currentMemoryUsage()
    }

    /// Shared by the public `getMemoryUsage()` override and the pre-flight
    /// guard in `HybridLiteRTLM+Execute.swift` — both need the same real,
    /// OS-level reading, not an estimate. `static` since it needs no
    /// instance state, which also makes it trivially callable from the
    /// extension file.
    static func currentMemoryUsage() -> MemoryUsage {
        var residentBytes: Double = 0.0
        var nativeHeapBytes: Double = 0.0

        // Retrieve process resident set size (RSS) via Mach basic task info
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size / MemoryLayout<integer_t>.size)
        let kerr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }

        if kerr == KERN_SUCCESS {
            residentBytes = Double(info.resident_size)
            // iOS has no cheap native-heap figure equivalent to Android's
            // Debug.getNativeHeapAllocatedSize, so this aliases RSS. The two
            // fields are identical on iOS by design, not by accident.
            nativeHeapBytes = Double(info.resident_size)
        }

        // os_proc_available_memory reports headroom before Jetsam termination
        // (iOS 13+). Jetsam exists only on a real device; the Simulator is an
        // ordinary macOS process and gets 0.
        let availableBytes = Double(os_proc_available_memory())

        return MemoryUsage(
            nativeHeapBytes: nativeHeapBytes,
            residentBytes: residentBytes,
            availableMemoryBytes: availableBytes,
            isLowMemory: isLowMemory(availableBytes: availableBytes)
        )
    }

    /// Low memory means under ~200MB of Jetsam headroom.
    ///
    /// A zero reading is the absence of a measurement, not the absence of
    /// memory: read as pressure, it rejected every execute() on the Simulator
    /// with 507. Treated as unknown instead, which is what Android does when
    /// ActivityManager is unreachable and what the JS headroom check does with
    /// the same zero. The cost is no memory guard where nothing can be
    /// measured, which is right for a Simulator.
    static func isLowMemory(availableBytes: Double) -> Bool {
        return availableBytes > 0 && availableBytes < 200.0 * 1024.0 * 1024.0
    }
    
    public func checkModelCapabilities(modelPath: String) throws -> ModelCapabilities {
        // iOS LiteRT-LM C API doesn't expose a Capabilities class like Android.
        // Return safe defaults — speculative decoding support can be detected
        // at engine init time on iOS.
        return ModelCapabilities(supportsSpeculativeDecoding: false)
    }

    public func getActiveBackend() throws -> Backend {
        return backend
    }

    public func stopGeneration() throws {
        queue.async {
            guard let conversation = self.conversation else { return }
            litert_lm_conversation_cancel_process(conversation)
            NSLog("[LiteRTLM] stopGeneration: cancelled active inference")
        }
    }

    public func close() throws {
        queue.sync {
            closeInternal()
        }
    }
    
    // MARK: - Async Operations
    
    public func loadModel(modelPath: String, config: LLMConfig?) throws -> Promise<Void> {
        let promise = Promise<Void>()
        
        queue.async {
            // Teardown any previous contexts
            self.closeInternal()
            
            // Extract configurations
            if let config = config {
                if let b = config.backend { self.backend = b }
                if let t = config.temperature { self.temperature = t }
                if let k = config.topK { self.topK = Int(k) }
                if let p = config.topP { self.topP = p }
                // New split fields take priority over legacy maxTokens
                if let ctx = config.maxContextTokens { self.maxContextTokens = Int(ctx) }
                if let out = config.maxOutputTokens { self.maxOutputTokens = Int(out) }
                // Legacy: if only maxTokens is set, map to both for backward compat
                if config.maxContextTokens == nil && config.maxOutputTokens == nil,
                   let m = config.maxTokens {
                    self.maxContextTokens = Int(m)
                    self.maxOutputTokens = Int(m)
                }
                if let s = config.systemPrompt { self.systemPrompt = s }
                self.tools = config.tools
                self.enableSpeculativeDecoding = config.enableSpeculativeDecoding ?? false
                self.enableThinking = config.enableThinking ?? false
            } else {
                self.tools = nil
                self.enableSpeculativeDecoding = false
                self.enableThinking = false
            }
            
            // Map main backend string
            let mainBackendStr = self.backend == .gpu ? "gpu" : (self.backend == .npu ? "gpu" : "cpu")
            
            //Sniff multimodal support
            let isMultimodal = config?.multimodal ?? (modelPath.lowercased().contains("3n") || modelPath.lowercased().contains("gemma3"))
            let visionBackend = isMultimodal ? "gpu" : nil
            let audioBackend = isMultimodal ? "cpu" : nil
            
            var rawEngine: OpaquePointer? = nil
            
            // Set LiteRT C Log Level to WARNING (2) for clean production output
            // v0.15: takes a typed LiteRtLmLogSeverity (previously a raw int,
            // where 2 was INFO — WARNING is the intended level).
            litert_lm_set_min_log_level(kLiteRtLmLogSeverityWarning)
            
            // Creation helper with scoped FFI pointer lifetime
            let createEngine = { (main: String, vision: String?, audio: String?) -> OpaquePointer? in
                let settings = modelPath.withCString { modelC in
                    self.withOptionalCString(main) { mainC in
                        self.withOptionalCString(vision) { visionC in
                            self.withOptionalCString(audio) { audioC in
                                return litert_lm_engine_settings_create(modelC, mainC, visionC, audioC)
                            }
                        }
                    }
                }
                
                guard let s = settings else { return nil }
                defer { litert_lm_engine_settings_delete(s) }
                
                litert_lm_engine_settings_set_max_num_tokens(s, Int32(self.maxContextTokens))
                litert_lm_engine_settings_enable_benchmark(s)
                
                if self.enableSpeculativeDecoding {
                    if let loadedFile = litert_lm_loaded_file_create((modelPath as NSString).utf8String) {
                        let hasMtp = litert_lm_loaded_file_has_speculative_decoding_support(loadedFile)
                        litert_lm_loaded_file_delete(loadedFile)
                        if hasMtp {
                            litert_lm_engine_settings_set_enable_speculative_decoding(s, true)
                        }
                    }
                }
                
                // Cache dir set to parent directory of model path
                let cacheDir = (modelPath as NSString).deletingLastPathComponent
                cacheDir.withCString { cacheC in
                    litert_lm_engine_settings_set_cache_dir(s, cacheC)
                }
                
                return litert_lm_engine_create(s)
            }
            
            // Attempt primary backend configuration
            rawEngine = createEngine(mainBackendStr, visionBackend, audioBackend)
            
            // Fallback sequence if GPU/NPU fails to initialize
            if rawEngine == nil {
                if mainBackendStr != "cpu" {
                    NSLog("[LiteRTLM] %@ backend failed — trying fallback chain...", mainBackendStr.uppercased())
                }
                // Fallback 1: CPU execution with GPU acceleration for heavy Vision parameters
                rawEngine = createEngine("cpu", "gpu", "cpu")
                
                if rawEngine == nil {
                    // Fallback 2: Full CPU execution for all modalities
                    rawEngine = createEngine("cpu", "cpu", "cpu")
                }
                
                if rawEngine == nil {
                    // Fallback 3: Text-only CPU execution (skip vision executor mapping)
                    rawEngine = createEngine("cpu", nil, nil)
                }
                
                if rawEngine != nil {
                    NSLog("[LiteRTLM] %@ backend unavailable — fell back to CPU successfully", mainBackendStr.uppercased())
                    self.backend = .cpu
                }
            }
            
            guard let engine = rawEngine else {
                promise.reject(withError: NSError(domain: "LiteRTLM", code: 500, userInfo: [NSLocalizedDescriptionKey: "Failed to construct LiteRT-LM engine. Checked backends and fallback chains."]))
                return
            }
            
            self.engine = engine
            self.createNewConversation()
            self.loadedModelPath = modelPath
            
            guard self.conversation != nil else {
                self.closeInternal()
                promise.reject(withError: NSError(domain: "LiteRTLM", code: 500, userInfo: [NSLocalizedDescriptionKey: "Failed to create conversation context."]))
                return
            }
            
            self.isLoaded = true
            promise.resolve()
        }
        
        return promise
    }
    
    // Legacy inference — shapes mirror src/inferenceRouting.ts; JS createLLM routes via execute.
    public func sendMessage(message: String) throws -> Promise<ExecuteResult> {
        try execute(parts: [.textPart(message)], onToken: nil)
    }

    public func sendMessageAsync(
        message: String,
        onToken: @escaping (_ token: String, _ done: Bool) -> Void
    ) throws -> Promise<Void> {
        try executeVoid(parts: [.textPart(message)], onToken: onToken)
    }

    public func sendMessageWithImage(message: String, imagePath: String) throws -> Promise<ExecuteResult> {
        try execute(parts: [.textPart(message), .imagePart(imagePath)], onToken: nil)
    }

    public func sendMessageWithImageAsync(
        message: String, imagePath: String,
        onToken: @escaping (_ token: String, _ done: Bool) -> Void
    ) throws -> Promise<Void> {
        try executeVoid(parts: [.textPart(message), .imagePart(imagePath)], onToken: onToken)
    }

    public func sendMessageWithAudioAsync(
        message: String, audioPath: String,
        onToken: @escaping (_ token: String, _ done: Bool) -> Void
    ) throws -> Promise<Void> {
        try executeVoid(parts: [.textPart(message), .audioPart(audioPath)], onToken: onToken)
    }

    public func sendMessageWithAudio(message: String, audioPath: String) throws -> Promise<ExecuteResult> {
        try execute(parts: [.textPart(message), .audioPart(audioPath)], onToken: nil)
    }

    public func sendMultimodalMessage(parts: [MultimodalPart]) throws -> Promise<ExecuteResult> {
        try execute(parts: parts, onToken: nil)
    }

    public func sendToolResponse(
        responses: [ToolResponse],
        onToken: ((_ token: String, _ done: Bool) -> Void)?
    ) throws -> Promise<ExecuteResult> {
        // Format tool results as a message and send to the conversation
        let toolResultText = responses.map { "Tool '\($0.name)' result: \($0.responseJson)" }
            .joined(separator: "\n")
        return try execute(parts: [.textPart(toolResultText)], onToken: onToken)
    }

    public func downloadModel(
        url: String,
        fileName: String,
        onProgress: ((Double) -> Void)?
    ) throws -> Promise<String> {
        return try modelStore.downloadFile(
            url: url,
            fileName: fileName,
            headersJson: "{}",
            onProgress: { progress in
                onProgress?(progress)
            }
        )
    }
    
    public func deleteModel(fileName: String) throws -> Promise<Void> {
        let promise = Promise<Void>()
        
        queue.async {
            do {
                try self.modelStore.deleteFile(fileName: fileName)
                let currentlyLoadedName = self.loadedModelPath.map { ($0 as NSString).lastPathComponent.lowercased() }
                if let loadedName = currentlyLoadedName, loadedName == fileName.lowercased() {
                    if self.isLoaded {
                        self.closeInternal()
                    }
                }
                promise.resolve()
            } catch {
                promise.reject(withError: error)
            }
        }
        
        return promise
    }
    
    // MARK: - Internal Engine Helpers
    
    private func createNewConversation(initialMessages: [Message] = []) {
        guard let engine = self.engine else { return }
        
        if let oldConv = self.conversation {
            litert_lm_conversation_delete(oldConv)
            self.conversation = nil
        }
        
        guard let convConfig = litert_lm_conversation_config_create() else { return }
        defer { litert_lm_conversation_config_delete(convConfig) }
        
        guard let sessionConfig = litert_lm_session_config_create() else { return }
        defer { litert_lm_session_config_delete(sessionConfig) }
        
        litert_lm_session_config_set_max_output_tokens(sessionConfig, Int32(self.maxOutputTokens))
        
        // v0.15: sampler params are opaque — built through create/setters
        // instead of a stack struct. set_sampler_params copies the values into
        // the session config, so the params can be freed when this scope ends.
        guard let sampler = litert_lm_sampler_params_create(kLiteRtLmSamplerTypeTopP) else { return }
        defer { litert_lm_sampler_params_delete(sampler) }
        litert_lm_sampler_params_set_top_k(sampler, Int32(self.topK))
        litert_lm_sampler_params_set_top_p(sampler, Float(self.topP))
        litert_lm_sampler_params_set_temperature(sampler, Float(self.temperature))
        litert_lm_sampler_params_set_seed(sampler, 0)
        litert_lm_session_config_set_sampler_params(sessionConfig, sampler)
        
        litert_lm_conversation_config_set_session_config(convConfig, sessionConfig)

        // Always stated, never omitted, for the reason Android passes
        // `enable_thinking` on every message: chat templates test it as
        // "defined and false", so leaving it out lets a reasoning model reason
        // whatever the setting says. The engine writes it into the template
        // context as a real boolean, and when on, routes reasoning to the
        // model's thinking channel (`channels` in each message). The value is
        // copied into the conversation config, so this can be freed after.
        if let thinkingConfig = litert_lm_thinking_config_create() {
            defer { litert_lm_thinking_config_delete(thinkingConfig) }
            litert_lm_thinking_config_set_enable_thinking(thinkingConfig, self.enableThinking)
            litert_lm_conversation_config_set_thinking_config(convConfig, thinkingConfig)
        }

        if let systemPrompt = self.systemPrompt {
            let systemMsgJson = "{\"role\":\"system\",\"content\":\"" + escapeJson(systemPrompt) + "\"}"
            systemMsgJson.withCString { systemMsgC in
                litert_lm_conversation_config_set_system_message(convConfig, systemMsgC)
            }
        }
        
        if let tools = self.tools, !tools.isEmpty {
            var toolsArray: [[String: Any]] = []
            for tool in tools {
                var functionMap: [String: Any] = ["name": tool.name, "description": tool.description]
                if let data = tool.parametersJson.data(using: .utf8),
                   let parsedParams = try? JSONSerialization.jsonObject(with: data, options: []) {
                    functionMap["parameters"] = parsedParams
                }
                toolsArray.append(["type": "function", "function": functionMap])
            }
            if let data = try? JSONSerialization.data(withJSONObject: toolsArray, options: []),
               let jsonString = String(data: data, encoding: .utf8) {
                jsonString.withCString { toolsC in
                    litert_lm_conversation_config_set_tools(convConfig, toolsC)
                }
            }
        }
        
        if !initialMessages.isEmpty {
            let payload = initialMessages.map { msg -> [String: String] in
                let role: String
                switch msg.role {
                case .model: role = "model"
                case .system: role = "system"
                default: role = "user"
                }
                return ["role": role, "content": msg.content]
            }
            if let data = try? JSONSerialization.data(withJSONObject: payload, options: []),
               let jsonString = String(data: data, encoding: .utf8) {
                jsonString.withCString { messagesC in
                    litert_lm_conversation_config_set_messages(convConfig, messagesC)
                }
            }
        }

        self.conversation = litert_lm_conversation_create(engine, convConfig)
    }
    
    private func closeInternal() {
        isLoaded = false
        history.removeAll()
        loadedModelPath = nil
        
        if let conversation = self.conversation {
            litert_lm_conversation_delete(conversation)
            self.conversation = nil
        }
        if let engine = self.engine {
            litert_lm_engine_delete(engine)
            self.engine = nil
        }
        
        lastStats = GenerationStats(
            promptTokens: 0.0,
            completionTokens: 0.0,
            totalTokens: 0.0,
            timeToFirstToken: 0.0,
            totalTime: 0.0,
            tokensPerSecond: 0.0
        )
    }
    
    // MARK: - Internal Preprocessing Helpers (for extension files)
    
    private let kControlTokens = [
        "<end_of_turn>",
        "<start_of_turn>model",
        "<start_of_turn>user",
        "<start_of_turn>",
        "<eos>"
    ]
    
    func escapeJson(_ input: String) -> String {
        var output = ""
        for char in input {
            switch char {
            case "\"": output += "\\\""
            case "\\": output += "\\\\"
            case "\n": output += "\\n"
            case "\r": output += "\\r"
            case "\t": output += "\\t"
            case "\u{0008}": output += "\\b"
            case "\u{000c}": output += "\\f"
            default: output.append(char)
            }
        }
        return output
    }
    
    func stripControlTokens(_ text: String) -> String {
        var result = text
        for tok in kControlTokens {
            result = result.replacingOccurrences(of: tok, with: "")
        }
        return result
    }
    
    func safeEmitLength(_ text: String) -> Int {
        let chars = Array(text)
        guard let lastAngleIdx = chars.lastIndex(of: "<") else {
            return chars.count
        }
        let suffix = String(chars[lastAngleIdx...])
        for tok in kControlTokens {
            if tok.hasPrefix(suffix) && suffix.count < tok.count {
                return lastAngleIdx
            }
        }
        return chars.count
    }
    
    /// One engine message, split into the three things it can carry.
    struct EngineMessage {
        var text = ""
        var toolCalls: [ToolCall] = []
        var thinking = ""
    }

    /// Reads a message the C API returned, whole or as one streamed chunk.
    ///
    /// The engine sends JSON of three shapes, often one per chunk:
    ///   text      {"role":"assistant","content":[{"type":"text","text":"…"}]}
    ///   thinking  {"role":"assistant","channels":{"thought":"…"}}
    ///   tool call {"role":"assistant","tool_calls":[{"type":"function",
    ///              "function":{"name":"…","arguments":{…}}}]}
    ///
    /// Only `content` used to be read, and a message without it fell back to
    /// its raw JSON as text — so every tool call was shown to the reader as
    /// JSON instead of reaching JS on `toolCalls`, and thinking was lost. This
    /// mirrors what the Kotlin SDK's `jsonToMessage` gives Android.
    ///
    /// Control tokens are left in `text`: a streamed token can be split across
    /// chunks, so callers strip them from the accumulated text.
    func parseEngineMessage(_ raw: String) -> EngineMessage {
        var message = EngineMessage()
        guard let data = raw.data(using: .utf8),
              let json = (try? JSONSerialization.jsonObject(with: data, options: [])) as? [String: Any] else {
            // Not a message object: plain text.
            message.text = raw
            return message
        }

        if let content = json["content"] as? String {
            message.text = content
        } else if let parts = json["content"] as? [[String: Any]] {
            for part in parts where (part["type"] as? String) == "text" {
                message.text += (part["text"] as? String) ?? ""
            }
        }

        if let calls = json["tool_calls"] as? [[String: Any]] {
            for call in calls {
                guard let function = call["function"] as? [String: Any],
                      let name = function["name"] as? String else { continue }
                message.toolCalls.append(
                    ToolCall(name: name, argumentsJson: argumentsJson(function["arguments"])))
            }
        }

        // Every channel, not only "thought": channel names come from each
        // model's metadata, and upstream treats the first one as thinking.
        // Sorted so a message carrying several reads the same every time.
        if let channels = json["channels"] as? [String: Any] {
            for key in channels.keys.sorted() {
                if let text = channels[key] as? String { message.thinking += text }
            }
        }

        return message
    }

    /// Tool arguments as the JSON string JS expects. The engine sends an
    /// object; a string is passed through in case a model encodes it that way.
    private func argumentsJson(_ value: Any?) -> String {
        if let string = value as? String { return string }
        if let value = value,
           JSONSerialization.isValidJSONObject(value),
           let data = try? JSONSerialization.data(withJSONObject: value, options: []),
           let string = String(data: data, encoding: .utf8) {
            return string
        }
        return "{}"
    }
    
    private func withOptionalCString<R>(_ string: String?, _ block: (UnsafePointer<CChar>?) -> R) -> R {
        if let string = string {
            return string.withCString { block($0) }
        } else {
            return block(nil)
        }
    }

}
