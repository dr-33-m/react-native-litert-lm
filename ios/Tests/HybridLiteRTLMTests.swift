import XCTest
@testable import LiteRTLM

class HybridLiteRTLMTests: XCTestCase {
    var bridge: HybridLiteRTLM!

    override func setUp() {
        super.setUp()
        bridge = HybridLiteRTLM()
    }

    override func tearDown() {
        try? bridge.close()
        bridge = nil
        super.tearDown()
    }

    func testPathTraversalRejection() async throws {
        let traversals = ["../../etc/passwd", "/absolute/path/file", "subdir\\..\\file", "..", "../", "..\\"]
        for traversal in traversals {
            do {
                let promise = try bridge.deleteModel(fileName: traversal)
                _ = try await promise.await()
                XCTFail("Should have failed for traversal: \(traversal)")
            } catch {
                let nsError = error as NSError
                XCTAssertEqual(nsError.domain, "LiteRTLM")
                XCTAssertEqual(nsError.code, 400)
                XCTAssertTrue(nsError.localizedDescription.contains("path traversal") || nsError.localizedDescription.contains("directory separators"))
            }
        }
    }

    func testNonHTTPSDownloadRejection() async throws {
        do {
            let promise = try bridge.downloadModel(url: "http://insecure-domain.com/model.bin", fileName: "model.bin", onProgress: nil)
            _ = try await promise.await()
            XCTFail("Should have blocked insecure HTTP downloads")
        } catch {
            let nsError = error as NSError
            XCTAssertEqual(nsError.domain, "LiteRTLM")
            XCTAssertEqual(nsError.code, 400)
            XCTAssertTrue(nsError.localizedDescription.contains("HTTPS is required"))
        }
    }

    func testMemoryTelemetry() {
        XCTAssertNoThrow(try bridge.getMemoryUsage())
        if let mem = try? bridge.getMemoryUsage() {
            XCTAssertGreaterThanOrEqual(mem.nativeHeapBytes, 0.0)
            XCTAssertGreaterThanOrEqual(mem.residentBytes, 0.0)
            XCTAssertGreaterThanOrEqual(mem.availableMemoryBytes, 0.0)
            // The Simulator reports 0 headroom; that must not read as pressure.
            if mem.availableMemoryBytes == 0 {
                XCTAssertFalse(mem.isLowMemory)
            }
        }
    }

    func testParseEngineMessageReadsText() {
        let msg = bridge.parseEngineMessage(
            #"{"role":"assistant","content":[{"type":"text","text":"Hello"}]}"#)
        XCTAssertEqual(msg.text, "Hello")
        XCTAssertTrue(msg.toolCalls.isEmpty)
        XCTAssertEqual(msg.thinking, "")
    }

    func testParseEngineMessageReadsToolCallsWithoutLeakingJson() {
        let msg = bridge.parseEngineMessage(
            #"{"role":"assistant","tool_calls":[{"type":"function","function":{"name":"list_collections","arguments":{}}}]}"#)
        XCTAssertEqual(msg.text, "")
        XCTAssertEqual(msg.toolCalls.count, 1)
        XCTAssertEqual(msg.toolCalls.first?.name, "list_collections")
        XCTAssertEqual(msg.toolCalls.first?.argumentsJson, "{}")
    }

    func testParseEngineMessageReadsThinkingChannel() {
        let msg = bridge.parseEngineMessage(
            #"{"role":"assistant","channels":{"thought":"Let me check."}}"#)
        XCTAssertEqual(msg.text, "")
        XCTAssertEqual(msg.thinking, "Let me check.")
    }

    func testParseEngineMessagePassesPlainTextThrough() {
        XCTAssertEqual(bridge.parseEngineMessage("plain words").text, "plain words")
    }

    func testLowMemoryTreatsZeroReadingAsUnknown() {
        let mb = 1024.0 * 1024.0
        XCTAssertFalse(HybridLiteRTLM.isLowMemory(availableBytes: 0))
        XCTAssertTrue(HybridLiteRTLM.isLowMemory(availableBytes: 100 * mb))
        XCTAssertFalse(HybridLiteRTLM.isLowMemory(availableBytes: 1024 * mb))
    }

    func testSendMessageAsyncRejectsWithoutModel() async throws {
        do {
            let promise = try bridge.sendMessageAsync(message: "hello") { _, _ in }
            _ = try await promise.await()
            XCTFail("Should have failed without model")
        } catch {
            let nsError = error as NSError
            XCTAssertEqual(nsError.domain, "LiteRTLM")
            XCTAssertEqual(nsError.code, 400)
        }
    }

    func testSendMessageWithImageAsyncRejectsWithoutModel() async throws {
        do {
            let promise = try bridge.sendMessageWithImageAsync(message: "hello", imagePath: "/tmp/image.jpg") { _, _ in }
            _ = try await promise.await()
            XCTFail("Should have failed without model")
        } catch {
            let nsError = error as NSError
            XCTAssertEqual(nsError.domain, "LiteRTLM")
            XCTAssertEqual(nsError.code, 400)
        }
    }

    func testSendMessageWithAudioAsyncRejectsWithoutModel() async throws {
        do {
            let promise = try bridge.sendMessageWithAudioAsync(message: "hello", audioPath: "/tmp/audio.wav") { _, _ in }
            _ = try await promise.await()
            XCTFail("Should have failed without model")
        } catch {
            let nsError = error as NSError
            XCTAssertEqual(nsError.domain, "LiteRTLM")
            XCTAssertEqual(nsError.code, 400)
        }
    }

    func testSendMessageWithImageAsyncRejectsFileNotFound() async throws {
        do {
            let promise = try bridge.sendMessageWithImageAsync(message: "hello", imagePath: "/nonexistent/image.jpg") { _, _ in }
            _ = try await promise.await()
            XCTFail("Should have failed without model")
        } catch {
            let nsError = error as NSError
            XCTAssertEqual(nsError.domain, "LiteRTLM")
        }
    }

    func testSendMessageWithAudioAsyncRejectsFileNotFound() async throws {
        do {
            let promise = try bridge.sendMessageWithAudioAsync(message: "hello", audioPath: "/nonexistent/audio.wav") { _, _ in }
            _ = try await promise.await()
            XCTFail("Should have failed without model")
        } catch {
            let nsError = error as NSError
            XCTAssertEqual(nsError.domain, "LiteRTLM")
        }
    }

    func testInitialStats() {
        XCTAssertNoThrow(try bridge.getStats())
        if let stats = try? bridge.getStats() {
            XCTAssertEqual(stats.promptTokens, 0.0)
            XCTAssertEqual(stats.completionTokens, 0.0)
            XCTAssertEqual(stats.totalTokens, 0.0)
            XCTAssertEqual(stats.timeToFirstToken, 0.0)
            XCTAssertEqual(stats.totalTime, 0.0)
            XCTAssertEqual(stats.tokensPerSecond, 0.0)
        }
    }

    func testDeleteModelCleanupLogic() async throws {
        bridge.loadedModelPath = "/path/to/my_loaded_model.litertlm"

        let promise1 = try bridge.deleteModel(fileName: "other_model.litertlm")
        _ = try await promise1.await()
        XCTAssertEqual(bridge.loadedModelPath, "/path/to/my_loaded_model.litertlm")

        let promise2 = try bridge.deleteModel(fileName: "my_loaded_model.litertlm")
        _ = try await promise2.await()
        XCTAssertNil(bridge.loadedModelPath)
    }

    func testExecutePathPrecedenceOverBuffer() async throws {
        let pathPart = MultimodalPart(
            type: .image,
            text: nil,
            path: "/nonexistent/precedence_test_image.jpg",
            imageBuffer: nil,
            audioBuffer: nil
        )
        
        do {
            let promise = try bridge.execute(parts: [pathPart], onToken: nil)
            _ = try await promise.await()
            XCTFail("Should have failed")
        } catch {
            let nsError = error as NSError
            XCTAssertTrue(nsError.localizedDescription.contains("file not found: /nonexistent/precedence_test_image.jpg"))
        }
    }

    func testExecuteTempFileCleanupOnError() async throws {
        let dummyData = Data([0, 1, 2, 3])
        let buffer = ArrayBuffer(dummyData)
        
        let bufferPart = MultimodalPart(
            type: .image,
            text: nil,
            path: nil,
            imageBuffer: buffer,
            audioBuffer: nil
        )
        
        let invalidPathPart = MultimodalPart(
            type: .image,
            text: nil,
            path: "/nonexistent/invalid_file_cleanup_test.jpg",
            imageBuffer: nil,
            audioBuffer: nil
        )
        
        do {
            let promise = try bridge.execute(parts: [bufferPart, invalidPathPart], onToken: nil)
            _ = try await promise.await()
            XCTFail("Should have failed")
        } catch {
            let tempDirAfter = try FileManager.default.contentsOfDirectory(atPath: NSTemporaryDirectory())
            let leakedFiles = tempDirAfter.filter { $0.contains("litert_buf_") }
            XCTAssertEqual(leakedFiles.count, 0)
        }
    }
}
