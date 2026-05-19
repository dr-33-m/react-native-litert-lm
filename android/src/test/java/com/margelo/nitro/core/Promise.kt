package com.margelo.nitro.core

import androidx.annotation.Keep
import com.facebook.proguard.annotations.DoNotStrip

@Keep
@DoNotStrip
class Promise<T> {
    companion object {
        @JvmStatic
        fun <T> parallel(block: () -> T): Promise<T> {
            val promise = Promise<T>()
            try {
                val result = block()
                promise.resolve(result)
            } catch (e: Throwable) {
                promise.reject(e)
            }
            return promise
        }
    }

    var result: T? = null
        private set
    var error: Throwable? = null
        private set
    var isCompleted = false
        private set
    private val callbacks = mutableListOf<(T?, Throwable?) -> Unit>()

    fun resolve(value: T) {
        synchronized(this) {
            result = value
            isCompleted = true
            callbacks.forEach { it(value, null) }
        }
    }

    fun reject(exception: Throwable) {
        synchronized(this) {
            error = exception
            isCompleted = true
            callbacks.forEach { it(null, exception) }
        }
    }
}
