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
    
    /// Dedicated background serial queue to protect the JSI/JS thread from blocking and deadlocks (User Rule #1).
    let queue = DispatchQueue(label: "dev.litert.engine", qos: .userInteractive)
    
    /// Opaque pointer to the LiteRT LM C Engine.
    private var engine: OpaquePointer?
    
    /// Opaque pointer to the active conversation state.
    var conversation: OpaquePointer?
    
    /// Thread-safe status flag.
    private var isLoaded = false
    
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
            nativeHeapBytes = Double(info.resident_size)
        }
        
        // os_proc_available_memory reports actual headroom available before Jetsam termination (iOS 13+)
        let availableBytes = Double(os_proc_available_memory())
        
        // Flag memory warning at ~200MB remaining headroom
        let isLowMemory = availableBytes < 200.0 * 1024.0 * 1024.0
        
        return MemoryUsage(
            nativeHeapBytes: nativeHeapBytes,
            residentBytes: residentBytes,
            availableMemoryBytes: availableBytes,
            isLowMemory: isLowMemory
        )
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
            } else {
                self.tools = nil
                self.enableSpeculativeDecoding = false
            }
            
            // Map main backend string
            let mainBackendStr = self.backend == .gpu ? "gpu" : (self.backend == .npu ? "gpu" : "cpu")
            
            //Sniff multimodal support
            let isMultimodal = config?.multimodal ?? (modelPath.lowercased().contains("3n") || modelPath.lowercased().contains("gemma3"))
            let visionBackend = isMultimodal ? "gpu" : nil
            let audioBackend = isMultimodal ? "cpu" : nil
            
            var rawEngine: OpaquePointer? = nil
            
            // Set LiteRT C Log Level to WARNING (2) for clean production output
            litert_lm_set_min_log_level(2)
            
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
    public func sendMessage(message: String) throws -> Promise<String> {
        try execute(parts: [.textPart(message)], onToken: nil)
    }

    public func sendMessageAsync(
        message: String,
        onToken: @escaping (_ token: String, _ done: Bool) -> Void
    ) throws -> Promise<Void> {
        try executeVoid(parts: [.textPart(message)], onToken: onToken)
    }

    public func sendMessageWithImage(message: String, imagePath: String) throws -> Promise<String> {
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

    public func sendMessageWithAudio(message: String, audioPath: String) throws -> Promise<String> {
        try execute(parts: [.textPart(message), .audioPart(audioPath)], onToken: nil)
    }

    public func sendMultimodalMessage(parts: [MultimodalPart]) throws -> Promise<String> {
        try execute(parts: parts, onToken: nil)
    }

    public func downloadModel(
        url: String,
        fileName: String,
        onProgress: ((Double) -> Void)?
    ) throws -> Promise<String> {
        let store = HybridModelStore()
        return try store.downloadFile(
            url: url,
            fileName: fileName,
            headersJson: "",
            onProgress: { progress in
                onProgress?(progress)
            }
        )
    }
    
    public func deleteModel(fileName: String) throws -> Promise<Void> {
        let promise = Promise<Void>()
        
        queue.async {
            do {
                let store = HybridModelStore()
                try store.deleteFile(fileName: fileName)
                if self.isLoaded {
                    self.closeInternal()
                }
                promise.resolve()
            } catch {
                promise.reject(withError: error)
            }
        }
        
        return promise
    }
    
    // MARK: - Internal Engine Helpers
    
    private func createNewConversation() {
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
        
        var sampler = LiteRtLmSamplerParams()
        sampler.type = kLiteRtLmSamplerTypeTopP
        sampler.top_k = Int32(self.topK)
        sampler.top_p = Float(self.topP)
        sampler.temperature = Float(self.temperature)
        sampler.seed = 0
        withUnsafePointer(to: &sampler) { samplerPtr in
            litert_lm_session_config_set_sampler_params(sessionConfig, samplerPtr)
        }
        
        litert_lm_conversation_config_set_session_config(convConfig, sessionConfig)
        
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
        
        self.conversation = litert_lm_conversation_create(engine, convConfig)
    }
    
    private func closeInternal() {
        isLoaded = false
        history.removeAll()
        
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
    
    // MARK: - String and JSON Preprocessing Helpers
    
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
    
    func extractTextFromResponse(_ jsonResponse: String) -> String {
        guard let data = jsonResponse.data(using: .utf8) else {
            return stripControlTokens(jsonResponse)
        }
        do {
            if let json = try JSONSerialization.jsonObject(with: data, options: []) as? [String: Any] {
                if let content = json["content"] {
                    if let contentString = content as? String {
                        return stripControlTokens(contentString)
                    } else if let contentArray = content as? [[String: Any]] {
                        var textResult = ""
                        for part in contentArray {
                            if let type = part["type"] as? String, type == "text", let text = part["text"] as? String {
                                textResult += text
                            }
                        }
                        return stripControlTokens(textResult)
                    }
                }
            }
        } catch {}
        return stripControlTokens(jsonResponse)
    }
    
    private func withOptionalCString<R>(_ string: String?, _ block: (UnsafePointer<CChar>?) -> R) -> R {
        if let string = string {
            return string.withCString { block($0) }
        } else {
            return block(nil)
        }
    }

}
