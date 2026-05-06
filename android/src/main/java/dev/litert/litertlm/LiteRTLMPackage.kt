package dev.litert.litertlm

import android.os.Build
import android.util.Log
import com.facebook.react.TurboReactPackage
import com.facebook.react.bridge.NativeModule
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.module.model.ReactModuleInfo
import com.facebook.react.module.model.ReactModuleInfoProvider


import com.margelo.nitro.dev.litert.litertlm.LiteRTLMOnLoad

class LiteRTLMPackage : TurboReactPackage() {
    companion object {
        private const val TAG = "LiteRTLMPackage"

        private fun isSupportedPrimaryAbi(): Boolean {
            val primaryAbi = Build.SUPPORTED_64_BIT_ABIS.firstOrNull() ?: return false
            return primaryAbi == "arm64-v8a"
        }
    }
    init {
        if (!isSupportedPrimaryAbi()) {
            Log.w(TAG, "Skipping LiteRTLM native init on unsupported primary ABI: ${Build.SUPPORTED_64_BIT_ABIS.firstOrNull()}")
        } else {
            try {
                LiteRTLMOnLoad.initializeNative()
            } catch (e: UnsatisfiedLinkError) {
                Log.e(TAG, "LiteRTLM native init failed; disabling LiteRTLM for this process.", e)
            }
        }
    }


    override fun getModule(name: String, reactContext: ReactApplicationContext): NativeModule? {
        return null
    }

    override fun getReactModuleInfoProvider(): ReactModuleInfoProvider {
        return ReactModuleInfoProvider { emptyMap<String, ReactModuleInfo>() }
    }
}
