///
/// HybridLiteRTLM.kt
/// Kotlin implementation of LiteRTLM HybridObject using LiteRT-LM Android SDK.
///

package com.margelo.nitro.dev.litert.litertlm

import android.util.Log
import android.os.Debug
import android.app.ActivityManager
import android.content.Context
import java.util.Collections
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicReference
import androidx.annotation.Keep
import com.facebook.proguard.annotations.DoNotStrip
import dev.litert.litertlm.LiteRTLMInitProvider
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.Conversation
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.SamplerConfig
import com.margelo.nitro.dev.litert.litertlm.Backend
import com.margelo.nitro.dev.litert.litertlm.GenerationStats
import com.margelo.nitro.dev.litert.litertlm.HybridLiteRTLMSpec
import com.margelo.nitro.dev.litert.litertlm.LLMConfig
import com.margelo.nitro.dev.litert.litertlm.Message
import com.margelo.nitro.dev.litert.litertlm.Role
import com.margelo.nitro.core.Promise
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.ExperimentalApi
import com.google.ai.edge.litertlm.ExperimentalFlags
import com.google.ai.edge.litertlm.OpenApiTool
import com.google.ai.edge.litertlm.ToolProvider
import com.google.ai.edge.litertlm.tool



// Alias to avoid confusion with our generated Message type
// Alias to avoid confusion
typealias LiteRTMessage = com.google.ai.edge.litertlm.Message



/**
 * Kotlin implementation of LiteRTLM using the LiteRT-LM Android SDK.
 * This class bridges between React Native (via Nitro) and the Google LiteRT-LM Engine.
 */
@DoNotStrip
@Keep
class HybridLiteRTLM : HybridLiteRTLMSpec() {

    companion object {
        private const val TAG = "HybridLiteRTLM"

        /** How long a teardown waits for a cancelled decode loop to stop. */
        private const val CANCEL_SETTLE_TIMEOUT_SECONDS = 2L
        private val initLock = Any()

        /** Cached result of OpenCL availability probe (null = not yet checked). */
        @Volatile
        private var openCLAvailable: Boolean? = null

        /** Cached result of NPU/QNN availability probe (null = not yet checked). */
        @Volatile
        private var npuAvailable: Boolean? = null
        
        /**
         * Initialize the native library.
         * Must be called from Application.onCreate() to register the HybridObject.
         */
        fun initialize() {
            try {
                // Call generated internal OnLoad to load the library
                LiteRTLMOnLoad.initializeNative()
            } catch (e: Throwable) {
                Log.e(TAG, "Failed to initialize LiteRTLM native library", e)
            }
        }
    }

    init {
        LiteRTLMRegistry.register(this)
    }

    // LiteRT-LM Engine and Conversation
    private var engine: Engine? = null
    private var conversation: Conversation? = null
    
    @Volatile
    private var isClosed = false

    /**
     * Latch of the streaming worker currently decoding, if any.
     *
     * Published so a teardown can wait for the decode loop to actually stop.
     * Closing a conversation out from under a running worker deletes native
     * state the worker is still reading, which is a crash the Kotlin side
     * cannot catch.
     */
    @Volatile
    private var activeStreamingLatch: CountDownLatch? = null

    private val modelStore = HybridModelStore()
    private var loadedModelPath: String? = null

    // Conversation history for getHistory()
    // Synchronized to prevent ConcurrentModificationException: history is
    // written from Promise.parallel workers and sendMessageAsync SDK callbacks,
    // and read from getHistory() which may be called from the JS thread.
    private val history: MutableList<Message> = Collections.synchronizedList(mutableListOf())

    // Tool calls captured during inference via ToolProvider.execute()
    private val pendingToolCalls: MutableList<ToolCall> = Collections.synchronizedList(mutableListOf())

    // Last generation stats
    private var lastStats = GenerationStats(
        promptTokens = 0.0,
        completionTokens = 0.0,
        totalTokens = 0.0,
        timeToFirstToken = 0.0,
        totalTime = 0.0,
        tokensPerSecond = 0.0
    )

    // Configuration
    private var backend: Backend = Backend.CPU
    private var temperature: Double = 0.7
    private var topK: Int = 40
    private var topP: Double = 0.95
    private var maxContextTokens: Int = 4096
    private var maxOutputTokens: Int = 1024
    private var systemPrompt: String? = null
    private var tools: Array<ToolDefinition>? = null
    private var enableSpeculativeDecoding: Boolean = false
    private var enableThinking: Boolean = false

    override val memorySize: Long
        get() = 1024L * 1024L * 1024L // ~1GB (models are large)

    // -------------------------------------------------------------------------
    // loadModel - Initialize LiteRT-LM Engine and Conversation
    // -------------------------------------------------------------------------
    override fun loadModel(modelPath: String, config: LLMConfig?): Promise<Unit> {
        return Promise.parallel {
            // Serialize initialization to prevent OOM from concurrent loads
            synchronized(initLock) {
                if (isClosed) {
                    throw RuntimeException("Cannot load model: LiteRTLM instance is closed")
                }
                
                Log.i(TAG, "loadModel: $modelPath")
    
                // Clean up existing resources
                // We call internal cleanup that doesn't set isClosed
                cleanupInternal()
    
                // Apply configuration
                config?.let { cfg ->
                    cfg.backend?.let { backend = it }
                    cfg.temperature?.let { temperature = it }
                    cfg.topK?.let { topK = it.toInt() }
                    cfg.topP?.let { topP = it }
                    // New split fields take priority over legacy maxTokens
                    cfg.maxContextTokens?.let { maxContextTokens = it.toInt() }
                    cfg.maxOutputTokens?.let { maxOutputTokens = it.toInt() }
                    // Legacy: if only maxTokens is set, map to both for backward compat
                    if (cfg.maxContextTokens == null && cfg.maxOutputTokens == null) {
                        cfg.maxTokens?.let {
                            maxContextTokens = it.toInt()
                            maxOutputTokens = it.toInt()
                        }
                    }
                    cfg.systemPrompt?.let { systemPrompt = it }
                    tools = cfg.tools
                    enableSpeculativeDecoding = cfg.enableSpeculativeDecoding ?: false
                    enableThinking = cfg.enableThinking ?: false
                }
    
                try {
                    // Early GPU hardware check: probe for OpenCL library.
                    // LiteRT-LM's GPU delegate requires OpenCL, which is absent on
                    // most Samsung/Qualcomm devices. Probe once and cache the result —
                    // needed for both GPU main backend and multimodal vision backend.
                    if (openCLAvailable == null) {
                        val hasOpenCL = openCLAvailable ?: run {
                            val result = try {
                                System.loadLibrary("OpenCL")
                                true
                            } catch (_: UnsatisfiedLinkError) {
                                val paths = arrayOf(
                                    "/vendor/lib64/libOpenCL.so",
                                    "/system/vendor/lib64/libOpenCL.so",
                                    "/vendor/lib/libOpenCL.so",
                                    "/system/lib64/libOpenCL.so"
                                )
                                var loaded = false
                                for (path in paths) {
                                    try {
                                        System.load(path)
                                        loaded = true
                                        break
                                    } catch (_: UnsatisfiedLinkError) {}
                                }
                                loaded
                            }
                            openCLAvailable = result
                            result
                        }
                        if (!hasOpenCL) {
                            Log.w(TAG, "OpenCL library not found — GPU backend will likely fail, fallback chain will attempt CPU")
                        } else {
                            Log.i(TAG, "OpenCL library found — GPU backend is available")
                        }
                    }

                    // NPU hardware check: probe for QNN HTP runtime libraries.
                    // LiteRT-LM's NPU delegate requires Qualcomm QNN HTP (Hexagon Tensor
                    // Processor) runtime. Without it, Engine() crashes with SIGSEGV
                    // that Kotlin's try/catch cannot intercept.
                    // Unlike GPU (which throws a catchable Java exception on failure),
                    // NPU failure is a native crash — so we MUST detect before attempting.
                    if (npuAvailable == null) {
                        val hasNpu = run {
                            // Check for QNN HTP libraries in system vendor paths.
                            // These are only present on devices with Qualcomm NPU support.
                            val qnnPaths = arrayOf(
                                "/vendor/lib64/libQnnHtp.so",
                                "/vendor/lib/libQnnHtp.so",
                                "/system/vendor/lib64/libQnnHtp.so",
                                "/system/lib64/libQnnHtp.so",
                                "/vendor/lib64/libQnnSystem.so",
                                "/vendor/lib/libQnnSystem.so"
                            )
                            var found = false
                            for (path in qnnPaths) {
                                if (java.io.File(path).exists()) {
                                    found = true
                                    break
                                }
                            }
                            found
                        }
                        npuAvailable = hasNpu
                        if (!hasNpu) {
                            Log.w(TAG, "QNN HTP libraries not found — NPU backend unavailable")
                        } else {
                            Log.i(TAG, "QNN HTP libraries found — NPU backend may be available")
                        }
                    }

                    // Detect multimodal support. Check config.multimodal flag first, then fall back to filename sniffing.
                    // Only Gemma 3n bundles vision/audio executors; Gemma 4 E2B is text-only.
                    // Passing vision/audio backends to a text-only model causes
                    // vision_litert_compiled_model_executor init failures.
                    //
                    // "gemma3" used to be on this list and was wrong: Gemma 3 1B
                    // is text-only, and matching it here attached vision and
                    // audio executors that the model has no weights for. Only
                    // Gemma 3n is multimodal, and "3n" already catches it.
                    val modelFileName = modelPath.substringAfterLast("/").lowercase()
                    val isMultimodal = config?.multimodal ?: modelFileName.contains("3n")
    
                    // Get cache directory from application context
                    val cacheDirectory = LiteRTLMInitProvider.applicationContext?.cacheDir?.absolutePath
                    Log.i(TAG, "Using cache directory: $cacheDirectory")

                    if (enableSpeculativeDecoding) {
                        @OptIn(ExperimentalApi::class)
                        ExperimentalFlags.enableSpeculativeDecoding = true
                    }

                    // The most recent engine failure, kept so the thrown error can say
                    // what went wrong instead of only that everything was tried.
                    var lastEngineError: String? = null

                    // Helper: attempt engine creation with given backends, return null on failure
                    fun tryCreateEngine(
                        mainBackend: com.google.ai.edge.litertlm.Backend,
                        visionBackend: com.google.ai.edge.litertlm.Backend?,
                        audioBackend: com.google.ai.edge.litertlm.Backend?
                    ): Engine? {
                        return try {
                            val cfg = if (visionBackend != null && audioBackend != null) {
                                EngineConfig(
                                    modelPath = modelPath,
                                    backend = mainBackend,
                                    visionBackend = visionBackend,
                                    audioBackend = audioBackend,
                                    maxNumTokens = maxContextTokens,
                                    cacheDir = cacheDirectory
                                )
                            } else {
                                EngineConfig(
                                    modelPath = modelPath,
                                    backend = mainBackend,
                                    maxNumTokens = maxContextTokens,
                                    cacheDir = cacheDirectory
                                )
                            }
                            Engine(cfg).also { it.initialize() }
                        } catch (e: Exception) {
                            // Held onto because it is the only description of what
                            // actually went wrong. Reporting only "tried everything"
                            // turned a precise engine error ("Invalid magic number")
                            // into an unactionable one by the time it reached JS.
                            lastEngineError = e.message
                            Log.w(TAG, "Engine creation failed with backend $mainBackend: ${e.message}")
                            null
                        }
                    }

                    // Map our Backend enum to LiteRT-LM Backend sealed class.
                    // If hardware is unavailable, skip directly to CPU to avoid native
                    // crashes (SIGSEGV) that Kotlin's try/catch cannot intercept.
                    val lmBackend = when (backend) {
                        Backend.GPU -> {
                            val hasOpenCL = openCLAvailable ?: false
                            if (hasOpenCL) {
                                com.google.ai.edge.litertlm.Backend.GPU()
                            } else {
                                Log.w(TAG, "GPU requested but OpenCL unavailable — using CPU directly")
                                backend = Backend.CPU
                                com.google.ai.edge.litertlm.Backend.CPU()
                            }
                        }
                        Backend.NPU -> {
                            val hasNpu = npuAvailable ?: false
                            val nativeLibDir = LiteRTLMInitProvider.applicationContext?.applicationInfo?.nativeLibraryDir
                            Log.i(TAG, "NPU backend requested - available=$hasNpu, nativeLibraryDir=$nativeLibDir")
                            if (hasNpu && nativeLibDir != null) {
                                com.google.ai.edge.litertlm.Backend.NPU(nativeLibraryDir = nativeLibDir)
                            } else {
                                Log.w(TAG, "NPU requested but hardware unavailable — using CPU directly")
                                backend = Backend.CPU
                                com.google.ai.edge.litertlm.Backend.CPU()
                            }
                        }
                        else -> com.google.ai.edge.litertlm.Backend.CPU()
                    }

                    val lmVisionBackend = if (isMultimodal) {
                        if (openCLAvailable == true) com.google.ai.edge.litertlm.Backend.GPU()
                        else com.google.ai.edge.litertlm.Backend.CPU()
                    } else null
                    val lmAudioBackend = if (isMultimodal) com.google.ai.edge.litertlm.Backend.CPU() else null
    
                    Log.i(TAG, "Backend config: main=$lmBackend, vision=$lmVisionBackend, audio=$lmAudioBackend, multimodal=$isMultimodal")
    
                    if (isClosed) return@synchronized

                    // Engine configurations to try, each dropping one more
                    // capability than the last and ending at plain text-only CPU.
                    //
                    // The text-only row matters even when CPU was the backend
                    // asked for. A text-only model that gets sniffed as
                    // multimodal fails with vision and audio executors attached
                    // and loads the moment they are dropped, and gating this
                    // whole chain on "the caller wanted GPU or NPU" left the
                    // default CPU path with no recovery at all: one failed
                    // attempt and straight to an error.
                    fun cpu() = com.google.ai.edge.litertlm.Backend.CPU()
                    val attempts = mutableListOf(
                        Triple(lmBackend, lmVisionBackend, lmAudioBackend)
                    )
                    if (isMultimodal) {
                        attempts += Triple(cpu(), com.google.ai.edge.litertlm.Backend.GPU(), cpu())
                        attempts += Triple(cpu(), cpu(), cpu())
                    }
                    attempts += Triple(cpu(), null, null)

                    var eng: Engine? = null
                    for ((index, attempt) in attempts.withIndex()) {
                        if (isClosed) return@synchronized
                        val (main, vision, audio) = attempt
                        eng = tryCreateEngine(main, vision, audio)
                        if (eng == null) continue
                        if (index > 0) {
                            Log.w(TAG, "Primary backend failed; loaded on fallback #$index (main=$main, vision=$vision, audio=$audio)")
                            // Every fallback runs the main graph on CPU, so the
                            // reported active backend has to follow.
                            backend = Backend.CPU
                        }
                        break
                    }

                    engine = eng ?: throw RuntimeException(
                        "Failed to create LiteRT-LM engine after ${attempts.size} attempts: " +
                            (lastEngineError ?: "no error reported")
                    )
                    Log.i(TAG, "Engine created and initialized successfully")
    
                    // Create Conversation
                    createNewConversation()
                    Log.i(TAG, "Conversation created successfully")
                    loadedModelPath = modelPath
    
                } catch (e: Exception) {
                    Log.e(TAG, "Failed to load model: ${e.message}", e)
                    // Clean up partial state so isReady() returns false
                    cleanupInternal()
                    throw RuntimeException("Failed to load model: ${e.message}", e)
                }
            }
        }
    }

    // Legacy inference — shapes mirror src/inferenceRouting.ts; JS createLLM routes via execute.
    override fun sendMessage(message: String): Promise<ExecuteResult> =
        execute(parts = arrayOf(MultimodalPartFactories.textPart(message)), onToken = null)

    override fun sendMessageAsync(message: String, onToken: (String, Boolean) -> Unit): Promise<Unit> =
        executeVoid(parts = arrayOf(MultimodalPartFactories.textPart(message)), onToken = onToken)

    // -------------------------------------------------------------------------
    // Multimodal methods
    // -------------------------------------------------------------------------
    
    /**
     * Resolve non-filesystem URIs to real file paths.
     * Handles: asset:/// , file:///android_asset/ , content:// ,
     * drawable resource names, and raw resource names.
     */
    private fun resolveAssetUri(uri: String, ext: String): String {
        // Already a real filesystem path
        if (uri.startsWith("/") && java.io.File(uri).exists()) return uri
        if (uri.startsWith("file:///") && !uri.startsWith("file:///android_asset/")) {
            val path = uri.removePrefix("file://")
            if (java.io.File(path).exists()) return path
        }

        val context = LiteRTLMInitProvider.applicationContext
            ?: throw RuntimeException("Application context not available for asset resolution")
        val tempFile = java.io.File(context.cacheDir, "litert_asset_${java.util.UUID.randomUUID()}.$ext")

        try {
            // content:// URI (from image picker or some RN asset resolvers)
            if (uri.startsWith("content://")) {
                context.contentResolver.openInputStream(android.net.Uri.parse(uri))?.use { input ->
                    java.io.FileOutputStream(tempFile).use { output -> input.copyTo(output) }
                } ?: throw RuntimeException("Could not open content URI: $uri")
                return tempFile.absolutePath
            }

            // asset:/// or file:///android_asset/ — APK assets folder
            val assetName = when {
                uri.startsWith("asset:///") -> uri.removePrefix("asset:///")
                uri.startsWith("file:///android_asset/") -> uri.removePrefix("file:///android_asset/")
                else -> null
            }
            if (assetName != null) {
                context.assets.open(assetName).use { input ->
                    java.io.FileOutputStream(tempFile).use { output -> input.copyTo(output) }
                }
                return tempFile.absolutePath
            }

            // Plain name (e.g. "test") — React Native drawable/raw resource
            if (!uri.contains("/") && !uri.contains(":")) {
                // Try raw resource first (preserves original bytes — works for any format)
                val rawId = context.resources.getIdentifier(uri, "raw", context.packageName)
                if (rawId != 0) {
                    context.resources.openRawResource(rawId).use { input ->
                        java.io.FileOutputStream(tempFile).use { output -> input.copyTo(output) }
                    }
                    return tempFile.absolutePath
                }
                // Try drawable (images only — re-encodes as JPEG)
                val drawableId = context.resources.getIdentifier(uri, "drawable", context.packageName)
                if (drawableId != 0) {
                    val bitmap = android.graphics.BitmapFactory.decodeResource(context.resources, drawableId)
                    if (bitmap != null) {
                        try {
                            java.io.FileOutputStream(tempFile).use { out ->
                                bitmap.compress(android.graphics.Bitmap.CompressFormat.JPEG, 95, out)
                            }
                            return tempFile.absolutePath
                        } finally {
                            bitmap.recycle()
                        }
                    }
                }
            }
        } catch (e: Exception) {
            tempFile.delete()
            Log.e(TAG, "Failed to resolve asset URI '$uri': ${e.message}", e)
            throw RuntimeException("Failed to resolve media path: $uri", e)
        }

        return uri
    }

    /**
     * Resize image if dimensions exceed maxDimension to prevent OOM.
     * Gemma 3n's vision encoder is optimized for 512x512 or 1024x1024.
     * Passing larger images can spike memory 500MB+.
     */
    private fun resizeImageIfNeeded(imagePath: String, maxDimension: Int = 1024): String {
        val originalBitmap = android.graphics.BitmapFactory.decodeFile(imagePath)
            ?: throw RuntimeException("Failed to decode image: $imagePath")

        val width = originalBitmap.width
        val height = originalBitmap.height

        // If already within bounds, return original path
        if (width <= maxDimension && height <= maxDimension) {
            originalBitmap.recycle()
            return imagePath
        }

        Log.i(TAG, "Resizing image from ${width}x${height} to fit ${maxDimension}px")

        val scale = maxDimension.toFloat() / maxOf(width, height)
        val newWidth = (width * scale).toInt()
        val newHeight = (height * scale).toInt()

        val resizedBitmap = android.graphics.Bitmap.createScaledBitmap(originalBitmap, newWidth, newHeight, true)
        originalBitmap.recycle()

        // Save to temp file
        val cacheDir = LiteRTLMInitProvider.applicationContext?.cacheDir
            ?: throw RuntimeException("Application context not available for image resizing")
        val tempFile = java.io.File(cacheDir, "resized_${java.util.UUID.randomUUID()}.jpg")
        java.io.FileOutputStream(tempFile).use { out ->
            resizedBitmap.compress(android.graphics.Bitmap.CompressFormat.JPEG, 90, out)
        }
        resizedBitmap.recycle()

        Log.i(TAG, "Resized image saved to: ${tempFile.absolutePath} (${newWidth}x${newHeight})")
        return tempFile.absolutePath
    }

    override fun sendMessageWithImage(message: String, imagePath: String): Promise<ExecuteResult> =
        execute(
            parts = arrayOf(MultimodalPartFactories.textPart(message), MultimodalPartFactories.imagePart(imagePath)),
            onToken = null,
        )

    override fun sendMessageWithImageAsync(message: String, imagePath: String, onToken: (String, Boolean) -> Unit): Promise<Unit> =
        executeVoid(
            parts = arrayOf(MultimodalPartFactories.textPart(message), MultimodalPartFactories.imagePart(imagePath)),
            onToken = onToken,
        )

    override fun downloadModel(url: String, fileName: String, onProgress: ((Double) -> Unit)?): Promise<String> {
        return modelStore.downloadFile(url, fileName, "{}", onProgress ?: {})
    }

    override fun deleteModel(fileName: String): Promise<Unit> {
        return Promise.parallel {
            modelStore.deleteFile(fileName)
            val currentlyLoadedName = loadedModelPath?.substringAfterLast("/")?.lowercase()
            if (currentlyLoadedName != null && currentlyLoadedName == fileName.lowercase()) {
                if (engine != null) {
                    cleanupInternal()
                }
            }
        }
    }

    override fun sendMessageWithAudioAsync(message: String, audioPath: String, onToken: (String, Boolean) -> Unit): Promise<Unit> =
        executeVoid(
            parts = arrayOf(MultimodalPartFactories.textPart(message), MultimodalPartFactories.audioPart(audioPath)),
            onToken = onToken,
        )

    override fun sendMessageWithAudio(message: String, audioPath: String): Promise<ExecuteResult> =
        execute(
            parts = arrayOf(MultimodalPartFactories.textPart(message), MultimodalPartFactories.audioPart(audioPath)),
            onToken = null,
        )

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------
    override fun getHistory(): Array<Message> {
        // Synchronized list requires manual sync for iteration/copy
        synchronized(history) {
            return history.toTypedArray()
        }
    }

    override fun resetConversation() {
        // A reset that recreates the conversation under a running decode loop
        // deletes the native state that loop is still reading.
        cancelInFlightGeneration()
        synchronized(history) {
            history.clear()
        }
        createNewConversation()
    }

    override fun resetConversationWith(messages: Array<Message>) {
        val seed = messages.map { msg ->
            LiteRTMessage(
                role = when (msg.role) {
                    Role.MODEL -> com.google.ai.edge.litertlm.Role.MODEL
                    Role.SYSTEM -> com.google.ai.edge.litertlm.Role.SYSTEM
                    else -> com.google.ai.edge.litertlm.Role.USER
                },
                contents = Contents.of(Content.Text(msg.content))
            )
        }
        synchronized(history) {
            history.clear()
            history.addAll(messages.toList())
        }
        // Recreating the conversation is what actually frees the old KV cache;
        // the seed is replayed into the new one without triggering generation.
        createNewConversation(seed)
    }

    override fun isReady(): Boolean {
        return isLoaded_
    }
    
    // Property backing field for isReady check
    private val isLoaded_: Boolean
        get() = engine != null

    override fun getStats(): GenerationStats {
        return lastStats
    }

    /** Shared by the public [getMemoryUsage] override and the pre-flight guard
     * in [execute] — both need the same real, OS-level reading, not an
     * estimate. */
    private fun currentMemoryUsage(): MemoryUsage {
        // Native heap: allocated bytes from Debug APIs (most accurate for native allocations)
        val nativeHeapBytes = Debug.getNativeHeapAllocatedSize().toDouble()

        // Process RSS: read from /proc/self/status (VmRSS) in kB
        var residentBytes = 0.0
        try {
            java.io.File("/proc/self/status").forEachLine { line ->
                if (line.startsWith("VmRSS:")) {
                    val kb = line.substringAfter("VmRSS:").trim().split("\\s+".toRegex())[0].toDoubleOrNull()
                    if (kb != null) {
                        residentBytes = kb * 1024.0
                    }
                    return@forEachLine
                }
            }
        } catch (e: Exception) {
            Log.w(TAG, "Failed to read /proc/self/status: ${e.message}")
        }

        // Available memory and low-memory flag from ActivityManager
        var availableMemoryBytes = 0.0
        var isLowMemory = false
        try {
            val context = LiteRTLMInitProvider.applicationContext
            if (context != null) {
                val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
                val memInfo = ActivityManager.MemoryInfo()
                activityManager.getMemoryInfo(memInfo)
                availableMemoryBytes = memInfo.availMem.toDouble()
                isLowMemory = memInfo.lowMemory
            }
        } catch (e: Exception) {
            Log.w(TAG, "Failed to get ActivityManager memory info: ${e.message}")
        }

        return MemoryUsage(
            nativeHeapBytes = nativeHeapBytes,
            residentBytes = residentBytes,
            availableMemoryBytes = availableMemoryBytes,
            isLowMemory = isLowMemory
        )
    }

    override fun getMemoryUsage(): MemoryUsage = currentMemoryUsage()

    override fun checkModelCapabilities(modelPath: String): ModelCapabilities {
        var supportsSpeculativeDecoding = false
        try {
            com.google.ai.edge.litertlm.Capabilities(modelPath).use {
                supportsSpeculativeDecoding = it.hasSpeculativeDecodingSupport()
            }
        } catch (e: Exception) {
            Log.w(TAG, "checkModelCapabilities: failed to query capabilities: ${e.message}")
        }
        return ModelCapabilities(supportsSpeculativeDecoding = supportsSpeculativeDecoding)
    }

    override fun getActiveBackend(): Backend = backend

    override fun stopGeneration() {
        try {
            conversation?.cancelProcess()
            Log.d(TAG, "stopGeneration: cancelled active inference")
        } catch (e: Exception) {
            Log.w(TAG, "stopGeneration: ${e.message}")
        }
    }

    /**
     * Signal the native decode loop to stop, then wait for the streaming worker
     * to settle.
     *
     * `cancelProcess()` only asks; the worker keeps running until it notices.
     * Anything that then closes or recreates the conversation would be deleting
     * native state while that worker still reads it, so every such path waits
     * here first. The timeout is a backstop: a worker that never settles is
     * worth a warning and a teardown anyway, because the alternative is hanging
     * the caller forever.
     */
    private fun cancelInFlightGeneration() {
        val latch = activeStreamingLatch ?: return
        try {
            conversation?.cancelProcess()
            if (!latch.await(CANCEL_SETTLE_TIMEOUT_SECONDS, TimeUnit.SECONDS)) {
                Log.w(TAG, "cancelInFlightGeneration: worker did not settle in time")
            }
        } catch (e: Exception) {
            Log.w(TAG, "cancelInFlightGeneration: ${e.message}")
        }
    }

    /**
     * Free the engine and its KV cache while leaving this instance usable.
     *
     * The distinction from `close()` is the whole point: `close()` sets
     * `isClosed`, after which even `loadModel()` is refused, so using it to
     * answer memory pressure permanently retired an instance that JS still
     * held and believed in.
     */
    fun releaseUnderMemoryPressure() {
        // Nothing loaded is nothing to free, and saying otherwise turns one
        // emergency into a page of identical warnings.
        if (engine == null) return
        Log.w(TAG, "Releasing engine under memory pressure; instance stays reloadable")
        cleanupInternal()
    }

    override fun close() {
        Log.d(TAG, "Closing resources")
        isClosed = true
        cleanupInternal()
        // A closed instance can never load again, so it has no business being
        // woken by the next memory emergency.
        LiteRTLMRegistry.unregister(this)
    }

    private fun cleanupInternal() {
        // Outside the lock: the decode loop does not need initLock, and waiting
        // for it while holding the lock would block loadModel behind it.
        cancelInFlightGeneration()
        synchronized(initLock) {
            /*
             * Dropped first, closed second.
             *
             * These used to be nulled after their `close()` calls, inside one
             * try. A close that threw — and closing a conversation that has
             * just been cancelled is exactly when one does — skipped the rest
             * of the block and left a retired engine still referenced. Nothing
             * downstream could tell: `isReady()` reads `engine != null`, so the
             * instance went on reporting itself loaded, and the next
             * `resetConversation()` called `createConversation` on a closed
             * engine and threw "Engine is not initialized" from deep in the SDK.
             *
             * Letting go of the references first makes that impossible: after
             * this block the instance is unloaded whatever the native side did.
             */
            val retiredConversation = conversation
            val retiredEngine = engine
            conversation = null
            engine = null
            loadedModelPath = null

            try {
                retiredConversation?.close()
            } catch (e: Exception) {
                Log.w(TAG, "Error closing conversation: ${e.message}")
            }
            try {
                retiredEngine?.close()
            } catch (e: Exception) {
                Log.w(TAG, "Error closing engine: ${e.message}")
            }
        }
    }

    private fun ensureLoaded() {
        if (engine == null) {
            throw RuntimeException("LiteRTLM: No model loaded. Call loadModel() first.")
        }
    }

    private fun createNewConversation(initialMessages: List<LiteRTMessage> = emptyList()) {
        ensureLoaded()
        // v0.10.2 enforces single-session: close existing conversation first
        conversation?.let { oldConv ->
            try {
                oldConv.close()
            } catch (e: Exception) {
                Log.w(TAG, "Failed to close old conversation: ${e.message}")
            }
            conversation = null
        }
        // Map tools. The engine only parses calls; JS runs them (see `automaticToolCalling`).
        val lmTools: List<ToolProvider>? = tools?.map { toolDef ->
            val apiTool = object : OpenApiTool {
                override fun getToolDescriptionJsonString(): String {
                    // SDK expects full OpenAPI tool description with name, description, and parameters
                    val fullDesc = org.json.JSONObject()
                    fullDesc.put("name", toolDef.name)
                    fullDesc.put("description", toolDef.description)
                    fullDesc.put("parameters", org.json.JSONObject(toolDef.parametersJson))
                    return fullDesc.toString()
                }
                override fun execute(paramsJsonString: String): String {
                    // Never invoked: automatic tool calling is off, so tool
                    // execution happens on the JS side.
                    return "{}"
                }
            }
            tool(apiTool)
        }

        // Create conversation config. NPU backend does not support SamplerConfig
        // (matching Gallery app pattern — setting sampler params on NPU causes crashes).
        val convConfig = ConversationConfig(
            samplerConfig = if (backend == Backend.NPU) null else SamplerConfig(
                topK = topK,
                topP = topP.toDouble(),
                temperature = temperature.toDouble(),
            ),
            systemInstruction = systemPrompt?.let { Contents.of(Content.Text(it)) },
            initialMessages = initialMessages,
            tools = lmTools ?: emptyList(),
            /*
             * Off, ported from upstream (hung-yueh 2cc938a). The SDK default runs
             * each parsed call against `execute()` above, which was a stub, and
             * feeds that stub's reply straight back to the model in the same
             * turn. The model then answered the stub: prose claiming a result it
             * never had ("I have removed it…"), streamed to the reader, then
             * discarded once JS ran the real tool and generated again. A wasted
             * generation per call, and its tokens stayed in the KV cache.
             * With it off, calls arrive on `Message.toolCalls` instead.
             */
            automaticToolCalling = false,
            // Available since LiteRT-LM 0.15.0; before that the Kotlin SDK had no
            // way to express it and every Android reply ran to the engine default.
            maxOutputToken = maxOutputTokens,
        )

        //
        // Upstream is actively adding max_output_tokens across API surfaces:
        //   - C API:    PR #2470 (merged 2026-06-04)
        //   - Python:   PR #2476 (merged 2026-06-04)
        //   - OpenAI:   PR #2433 (in progress)
        //   - Kotlin:   Not yet available — track at https://github.com/google-ai-edge/LiteRT-LM
        //
        // Once the Kotlin SDK exposes this, wire it via ConversationConfig here.
        // Checked rather than forced: this runs from `resetConversation`, which
        // JS calls on paths that do not first ask whether anything is loaded.
        // A named error beats a null-pointer dereference from inside the SDK.
        val activeEngine = engine
            ?: throw RuntimeException("Cannot create a conversation: no model is loaded.")
        conversation = activeEngine.createConversation(convConfig)
    }



    override fun sendMultimodalMessage(parts: Array<MultimodalPart>): Promise<ExecuteResult> {
        return execute(parts = parts, onToken = null)
    }

    /** Streaming adapter for legacy `Promise<Unit>` APIs — all inference runs through [execute]. */
    private fun executeVoid(
        parts: Array<MultimodalPart>,
        onToken: (String, Boolean) -> Unit,
    ): Promise<Unit> {
        val voidPromise = Promise<Unit>()
        try {
            execute(parts, onToken)
                .then { _ -> voidPromise.resolve(Unit) }
                .catch { voidPromise.reject(it) }
        } catch (e: Throwable) {
            voidPromise.reject(e)
        }
        return voidPromise
    }

    private class PreprocessedPart(
        val type: PartType,
        val text: String?,
        val path: String?,
        val bytes: ByteArray?
    )

    override fun execute(parts: Array<MultimodalPart>, onToken: ((token: String, done: Boolean) -> Unit)?): Promise<ExecuteResult> {
        // Preprocess synchronously on the JS/JSI thread to safely extract JS buffer bytes
        val preprocessed = parts.map { part ->
            val bytes = when (part.type) {
                PartType.IMAGE -> part.imageBuffer?.let { buf ->
                    val javaBuf = buf.getBuffer(false)
                    val arr = ByteArray(javaBuf.remaining())
                    javaBuf.get(arr)
                    arr
                }
                PartType.AUDIO -> part.audioBuffer?.let { buf ->
                    val javaBuf = buf.getBuffer(false)
                    val arr = ByteArray(javaBuf.remaining())
                    javaBuf.get(arr)
                    arr
                }
                else -> null
            }
            PreprocessedPart(
                type = part.type,
                text = part.text,
                path = part.path,
                bytes = bytes
            )
        }

        return Promise.parallel {
            ensureLoaded()

            // Refuse rather than risk it: growing the KV-cache under real memory
            // pressure can fail at the native engine level in a way that never
            // surfaces as a catchable Kotlin exception — it takes the whole
            // process down. A plain RuntimeException here is a normal rejected
            // Promise for callers, same contract as any other execute() failure.
            if (currentMemoryUsage().isLowMemory) {
                throw RuntimeException("LiteRTLM: Device memory is critically low; refusing to continue generation.")
            }

            // Clear any previous tool calls before new inference
            pendingToolCalls.clear()

            val tempFiles = mutableListOf<java.io.File>()

            try {
                val contents = mutableListOf<Content>()
                var userTextRepresentation = ""

                for (part in preprocessed) {
                    when (part.type) {
                        PartType.TEXT -> part.text?.let {
                            contents.add(Content.Text(it))
                            userTextRepresentation += "$it "
                        }
                        PartType.IMAGE -> {
                            val imagePath = when {
                                part.path != null -> {
                                    val resolved = resolveAssetUri(part.path, "jpg")
                                    if (resolved != part.path) tempFiles.add(java.io.File(resolved))
                                    resolved
                                }
                                part.bytes != null -> {
                                    val tmp = java.io.File(
                                        LiteRTLMInitProvider.applicationContext!!.cacheDir,
                                        "litert_buf_${java.util.UUID.randomUUID()}.jpg"
                                    )
                                    tmp.writeBytes(part.bytes)
                                    tempFiles.add(tmp)
                                    tmp.absolutePath
                                }
                                else -> null
                            }
                            if (imagePath != null) {
                                val processedPath = resizeImageIfNeeded(imagePath)
                                if (processedPath != imagePath) tempFiles.add(java.io.File(processedPath))
                                contents.add(Content.ImageFile(processedPath))
                                userTextRepresentation += "[Image] "
                            }
                        }
                        PartType.AUDIO -> {
                            val audioPath = when {
                                part.path != null -> {
                                    val resolved = resolveAssetUri(part.path, "wav")
                                    if (resolved != part.path) tempFiles.add(java.io.File(resolved))
                                    resolved
                                }
                                part.bytes != null -> {
                                    val tmp = java.io.File(
                                        LiteRTLMInitProvider.applicationContext!!.cacheDir,
                                        "litert_buf_${java.util.UUID.randomUUID()}.wav"
                                    )
                                    tmp.writeBytes(part.bytes)
                                    tempFiles.add(tmp)
                                    tmp.absolutePath
                                }
                                else -> null
                            }
                            if (audioPath != null) {
                                contents.add(Content.AudioFile(audioPath))
                                userTextRepresentation += "[Audio] "
                            }
                        }
                    }
                }

                userTextRepresentation = userTextRepresentation.trim()
                history.add(Message(Role.USER, userTextRepresentation))

                val userMsg = LiteRTMessage.user(Contents.of(contents))

                // Always stated, never omitted.
                //
                // `enable_thinking` is a chat-template variable, not a Gemma
                // feature, and templates test it as "defined and false". Qwen3's
                // is exactly that shape, so leaving the key out is not the same
                // as setting it to false: undefined falls through to the
                // template's own default, which for a reasoning model is to
                // reason. Omitting it meant thinking could be turned on but
                // never off, and the reasoning arrived inline in the answer.
                val extraContext: Map<String, String> =
                    mapOf("enable_thinking" to enableThinking.toString())

                if (onToken != null) {
                    // ── Streaming path ────────────────────────────────────────────────
                    val latch = CountDownLatch(1)
                    activeStreamingLatch = latch
                    val errorRef = AtomicReference<Throwable?>(null)
                    val fullResponseBuilder = StringBuilder()
                    val thinkingBuilder = StringBuilder()

                    val listener = StreamingCallbackListener(
                        onToken = { token, done ->
                            onToken(token, done)
                            if (done) latch.countDown()
                        },
                        responseBuilder = fullResponseBuilder,
                        thinkingBuilder = thinkingBuilder,
                        history = history,
                        userMessage = userTextRepresentation,
                        onStatsReady = { stats -> lastStats = stats },
                        onFailure = { e -> errorRef.set(e) },
                        onToolCalls = { calls -> captureToolCalls(calls) }
                    )

                    try {
                        conversation!!.sendMessageAsync(message = userMsg, callback = listener, extraContext = extraContext)
                    } catch (e: Exception) {
                        Log.e(TAG, "execute streaming failed", e)
                        errorRef.set(e)
                        onToken("Error: ${e.message}", true)
                        latch.countDown()
                    }

                    try {
                        latch.await()
                    } finally {
                        activeStreamingLatch = null
                    }
                    errorRef.get()?.let { throw RuntimeException("execute streaming failed: ${it.message}", it) }
                    val capturedToolCalls = synchronized(pendingToolCalls) {
                        pendingToolCalls.toTypedArray().also { pendingToolCalls.clear() }
                    }
                    ExecuteResult(
                        text = fullResponseBuilder.toString(),
                        toolCalls = capturedToolCalls,
                        thinkingText = thinkingBuilder.toString()
                    )

                } else {
                    // ── Blocking path ─────────────────────────────────────────────────
                    val startTime = System.nanoTime()
                    val responseMsg = conversation!!.sendMessage(message = userMsg, extraContext = extraContext)
                    val elapsedMs = (System.nanoTime() - startTime) / 1_000_000.0

                    val response = responseMsg.contents.contents
                        .filterIsInstance<Content.Text>()
                        .joinToString("") { it.text }

                    val thinkingText = responseMsg.channels["thought"] ?: ""
                    captureToolCalls(responseMsg.toolCalls)

                    history.add(Message(Role.MODEL, response))

                    val promptTokens = userTextRepresentation.length / 4.0
                    val completionTokens = response.length / 4.0
                    lastStats = GenerationStats(
                        promptTokens = promptTokens,
                        completionTokens = completionTokens,
                        totalTokens = promptTokens + completionTokens,
                        timeToFirstToken = 0.0,
                        totalTime = elapsedMs,
                        tokensPerSecond = if (elapsedMs > 0) completionTokens / (elapsedMs / 1000.0) else 0.0
                    )
                    val capturedToolCalls = synchronized(pendingToolCalls) {
                        pendingToolCalls.toTypedArray().also { pendingToolCalls.clear() }
                    }
                    ExecuteResult(
                        text = response,
                        toolCalls = capturedToolCalls,
                        thinkingText = thinkingText
                    )
                }
            } finally {
                // Clean up all temp files created during this execute call
                for (f in tempFiles) {
                    try { f.delete() } catch (e: Exception) {
                        Log.w(TAG, "Failed to delete temp file: ${f.absolutePath}")
                    }
                }
            }
        }
    }

    /** Records SDK-parsed calls in the shape JS receives on `ExecuteResult.toolCalls`. */
    private fun captureToolCalls(calls: List<com.google.ai.edge.litertlm.ToolCall>?) {
        for (call in calls.orEmpty()) {
            val argumentsJson = org.json.JSONObject(call.arguments).toString()
            Log.d(TAG, "Tool called: ${call.name} with args: $argumentsJson")
            pendingToolCalls.add(ToolCall(name = call.name, argumentsJson = argumentsJson))
        }
    }

    override fun sendToolResponse(
        responses: Array<ToolResponse>,
        onToken: ((token: String, done: Boolean) -> Unit)?
    ): Promise<ExecuteResult> {
        // Format tool results as a message and send to the conversation
        val toolResultText = responses.joinToString("\n") { response ->
            "Tool '${response.name}' result: ${response.responseJson}"
        }
        return execute(
            parts = arrayOf(MultimodalPartFactories.textPart(toolResultText)),
            onToken = onToken
        )
    }

    override fun countTokens(text: String): Double {
        return -1.0
    }
}
