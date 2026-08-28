package com.example.edge_ai_smart_bank_transfers

import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel

class MainActivity : FlutterActivity() {

    companion object {
        init {
            System.loadLibrary("edge_ai_native")
        }
    }

    private external fun nativePing(): String

    private val channel = "edge_ai/native"

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)

        MethodChannel(
            flutterEngine.dartExecutor.binaryMessenger,
            channel
        ).setMethodCallHandler { call, result ->

            when (call.method) {
                "ping" -> {
                    try {
                        result.success(nativePing())
                    } catch (e: Exception) {
                        result.error(
                            "NATIVE_ERROR",
                            e.message,
                            null
                        )
                    }
                }

                else -> result.notImplemented()
            }
        }
    }
}