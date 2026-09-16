package com.margelo.nitro.dev.litert.litertlm

import android.content.ComponentCallbacks2
import java.util.Collections
import java.util.WeakHashMap
import android.util.Log

/**
 * Global registry to track active LiteRTLM instances.
 * Used for memory trimming and cleanup.
 */
object LiteRTLMRegistry {
    private const val TAG = "LiteRTLMRegistry"

    // Use WeakSet-like structure to prevent leaks
    private val instances = Collections.newSetFromMap(WeakHashMap<HybridLiteRTLM, Boolean>())

    fun register(instance: HybridLiteRTLM) {
        synchronized(instances) {
            instances.add(instance)
        }
    }

    /**
     * Drop a retired instance.
     *
     * The set holds weak keys, so a dead instance does leave eventually — but
     * only once GC gets to it, and until then it is still iterated and still
     * logged against. Every `loadModel()` mints a fresh HybridLiteRTLM, so a
     * few reloads were enough for one memory emergency to report five engines
     * released when only the last of them held anything. Closing is a definite
     * end, so it is a better moment to forget an instance than a collection
     * that may not have happened yet.
     */
    fun unregister(instance: HybridLiteRTLM) {
        synchronized(instances) {
            instances.remove(instance)
        }
    }

    /**
     * Whether a trim level is a real emergency worth dropping engines for.
     *
     * Trim levels are event codes, not a severity scale, so a `>=` comparison
     * is the wrong shape entirely: TRIM_MEMORY_UI_HIDDEN is 20 and fires every
     * time the screen locks or the reader presses home, which says nothing
     * about memory. Testing `level >= TRIM_MEMORY_RUNNING_LOW` (10) therefore
     * matched a screen lock and tore the engine down on it.
     *
     * Only two codes mean the process dies without intervention:
     * RUNNING_CRITICAL (foreground, memory critical) and COMPLETE (cached and
     * next in line to be killed).
     */
    fun isMemoryEmergency(level: Int): Boolean =
        level == ComponentCallbacks2.TRIM_MEMORY_RUNNING_CRITICAL ||
            level >= ComponentCallbacks2.TRIM_MEMORY_COMPLETE

    fun onTrimMemory(level: Int) {
        Log.w(TAG, "Memory emergency (level=$level). Releasing engines...")
        synchronized(instances) {
            // Release the heavy native resources but keep each instance
            // reloadable. close() would set isClosed, and an instance that
            // refuses loadModel() afterwards is worse than one holding memory:
            // JS still points at it and has no way to tell it has been retired.
            instances.forEach { it.releaseUnderMemoryPressure() }
        }
    }
}
