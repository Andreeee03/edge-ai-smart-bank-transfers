package com.example.edge_ai_smart_bank_transfers

import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel
import java.io.File

class MainActivity : FlutterActivity() {

    private val channel = "edge_ai/native"

    companion object {
        init {
            System.loadLibrary("edge_ai_native")
        }
    }

    private external fun nativeInit(nativeLibDir: String): String
    private external fun nativePing(): String
    private external fun nativeLoadModel(modelPath: String): String
    private external fun nativeCreateContext(): String
    private external fun nativeTokenize(prompt: String): String
    private external fun nativeDecode(prompt: String): String
    private external fun nativeSampleFirstToken(): String
    private external fun nativeGenerate(prompt: String): String

    @Volatile
    private var nativeInitStatus: String? = null

    private fun ensureNativeInitialized(): String {
        synchronized(this) {
            if (nativeInitStatus == null) {
                nativeInitStatus =
                    nativeInit(applicationInfo.nativeLibraryDir)
            }

            return nativeInitStatus!!
        }
    }

    override fun configureFlutterEngine(
        flutterEngine: FlutterEngine
    ) {
        super.configureFlutterEngine(flutterEngine)

        MethodChannel(
            flutterEngine.dartExecutor.binaryMessenger,
            channel
        ).setMethodCallHandler { call, result ->

            when (call.method) {

                "ping" -> {
                    try {
                        result.success(
                            "${ensureNativeInitialized()}\n${nativePing()}"
                        )
                    } catch (e: Exception) {
                        result.error("NATIVE_ERROR", e.message, null)
                    }
                }

                "loadModel" -> {

                    val modelName =
                        "LFM2-700M_GPTPlus-DS_Q5_K_M.gguf"

                    val modelFile =
                        File(filesDir, "models/$modelName")

                    if (!modelFile.exists()) {
                        result.error(
                            "MODEL_NOT_FOUND",
                            "Model not found: ${modelFile.absolutePath}",
                            null
                        )

                        return@setMethodCallHandler
                    }

                    Thread {
                        try {
                            ensureNativeInitialized()

                            val response =
                                nativeLoadModel(
                                    modelFile.absolutePath
                                )

                            runOnUiThread {
                                result.success(response)
                            }
                        } catch (e: Exception) {
                            runOnUiThread {
                                result.error(
                                    "MODEL_LOAD_ERROR",
                                    e.message,
                                    null
                                )
                            }
                        }
                    }.start()
                }

                "createContext" -> runNative(
                    result,
                    "CONTEXT_ERROR"
                ) {
                    nativeCreateContext()
                }

                "tokenize" -> {
                    val prompt =
                        call.argument<String>("prompt") ?: ""

                    runNative(
                        result,
                        "TOKENIZE_ERROR"
                    ) {
                        nativeTokenize(prompt)
                    }
                }

                "decode" -> {
                    val prompt =
                        call.argument<String>("prompt") ?: ""

                    runNative(
                        result,
                        "DECODE_ERROR"
                    ) {
                        nativeDecode(prompt)
                    }
                }

                "sampleFirstToken" -> runNative(
                    result,
                    "SAMPLING_ERROR"
                ) {
                    nativeSampleFirstToken()
                }

                "generate" -> {
                    val prompt =
                        call.argument<String>("prompt") ?: ""

                    runNative(
                        result,
                        "GENERATION_ERROR"
                    ) {
                        nativeGenerate(prompt)
                    }
                }

                else -> result.notImplemented()
            }
        }
    }

    private fun runNative(
        result: MethodChannel.Result,
        errorCode: String,
        action: () -> String
    ) {
        Thread {
            try {
                val response = action()

                runOnUiThread {
                    result.success(response)
                }
            } catch (e: Exception) {
                runOnUiThread {
                    result.error(
                        errorCode,
                        e.message,
                        null
                    )
                }
            }
        }.start()
    }
}
