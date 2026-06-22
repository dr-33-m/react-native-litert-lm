package com.margelo.nitro.dev.litert.litertlm

import android.util.Log
import com.google.ai.edge.litertlm.Content

/**
 * Named implementation of the LiteRT-LM MessageCallback for streaming inference.
 *
 * Accumulates response chunks, forwards tokens to JS, and appends the final
 * response to the conversation history.
 */
internal class StreamingCallbackListener(
    private val onToken: (String, Boolean) -> Unit,
    private val responseBuilder: StringBuilder,
    private val thinkingBuilder: StringBuilder = StringBuilder(),
    private val history: MutableList<Message>,
    private val userMessage: String,
    private val onStatsReady: (GenerationStats) -> Unit,
    private val onFailure: ((Throwable) -> Unit)? = null,
    private val onThinkingToken: ((String) -> Unit)? = null,
) : com.google.ai.edge.litertlm.MessageCallback {

    private val startTime = System.nanoTime()
    private var firstTokenTime = 0L
    private var tokenCount = 0

    override fun onMessage(message: com.google.ai.edge.litertlm.Message) {
        val chunk = message.contents.contents
            .filterIsInstance<Content.Text>()
            .joinToString("") { it.text }

        // Capture thinking from the "thought" channel
        val thinkingChunk = message.channels["thought"]
        if (!thinkingChunk.isNullOrEmpty()) {
            thinkingBuilder.append(thinkingChunk)
            onThinkingToken?.invoke(thinkingChunk)
        }

        if (firstTokenTime == 0L && chunk.isNotEmpty()) {
            firstTokenTime = System.nanoTime()
        }
        if (chunk.isNotEmpty()) {
            tokenCount++
        }

        onToken(chunk, false)

        if (chunk.isNotEmpty()) {
            responseBuilder.append(chunk)
        }
    }

    override fun onDone() {
        val fullResponse = responseBuilder.toString()
        history.add(Message(Role.MODEL, fullResponse))

        // Compute stats using heuristic token counts (~4 chars/token)
        val elapsedMs = (System.nanoTime() - startTime) / 1_000_000.0
        val ttftMs = if (firstTokenTime > 0) (firstTokenTime - startTime) / 1_000_000.0 else 0.0
        val promptTokens = userMessage.length / 4.0
        val completionTokens = fullResponse.length / 4.0
        onStatsReady(GenerationStats(
            promptTokens = promptTokens,
            completionTokens = completionTokens,
            totalTokens = promptTokens + completionTokens,
            timeToFirstToken = ttftMs,
            totalTime = elapsedMs,
            tokensPerSecond = if (elapsedMs > 0) completionTokens / (elapsedMs / 1000.0) else 0.0
        ))

        Log.d("StreamingCallbackListener", "Streaming done. Length: ${fullResponse.length}, TTFT: ${ttftMs.toLong()}ms, Total: ${elapsedMs.toLong()}ms")

        // Notify JS that streaming is done AFTER updating history and stats
        onToken("", true)
    }

    override fun onError(throwable: Throwable) {
        if (throwable is kotlinx.coroutines.CancellationException ||
            throwable.message?.contains("cancel", ignoreCase = true) == true) {
            // Graceful cancellation via stopGeneration() — treat as done with partial content
            Log.i("StreamingCallbackListener", "Inference cancelled, emitting partial result")
            onDone()
            return
        }
        Log.e("StreamingCallbackListener", "Async generation failed", throwable)
        onToken("Error: ${throwable.message}", true)
        onFailure?.invoke(throwable)
    }
}
