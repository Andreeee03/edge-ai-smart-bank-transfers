package com.example.edge_ai_smart_bank_transfers

import android.Manifest
import android.content.ContentUris
import android.content.pm.PackageManager
import android.provider.CalendarContract
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel
import java.io.File

class MainActivity : FlutterActivity() {

    private val channel = "edge_ai/native"

    private val calendarPermissionRequestCode = 2001

    private var pendingCalendarPermissionResult:
        MethodChannel.Result? = null

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
                        result.error(
                            "NATIVE_ERROR",
                            e.message,
                            null
                        )
                    }
                }
                "loadModel" -> {
                    val modelName =
                        "LFM2-700M_Claude-DS_Q5_K_M.gguf"

                    val modelUrl =
                        "https://github.com/Andreeee03/edge-ai-smart-bank-transfers/releases/download/model-v1.0.0/LFM2-700M_Claude-DS_Q5_K_M.gguf"

                    val modelFile =
                        File(
                            filesDir,
                            "models/$modelName"
                        )

                    Thread {
                        try {
                            ensureNativeInitialized()

                            if (!modelFile.exists()) {
                                modelFile.parentFile?.mkdirs()

                                val tempFile =
                                    File(
                                        modelFile.parentFile,
                                        "$modelName.tmp"
                                    )

                                if (tempFile.exists()) {
                                    tempFile.delete()
                                }

                                val connection =
                                    java.net.URL(modelUrl)
                                        .openConnection() as java.net.HttpURLConnection

                                connection.connectTimeout = 15000
                                connection.readTimeout = 30000
                                connection.instanceFollowRedirects = true

                                try {
                                    connection.connect()

                                    if (connection.responseCode !in 200..299) {
                                        throw IllegalStateException(
                                            "Model download failed with HTTP ${connection.responseCode}"
                                        )
                                    }

                                    val totalBytes =
                                        connection.contentLengthLong

                                    connection.inputStream.use { input ->
                                        tempFile.outputStream().use { output ->
                                            val buffer =
                                                ByteArray(1024 * 1024)

                                            var downloadedBytes = 0L
                                            var lastPercent = -1

                                            while (true) {
                                                val count =
                                                    input.read(buffer)

                                                if (count == -1) {
                                                    break
                                                }

                                                output.write(
                                                    buffer,
                                                    0,
                                                    count
                                                )

                                                downloadedBytes += count

                                                if (totalBytes > 0) {
                                                    val percent =
                                                        (
                                                            downloadedBytes *
                                                                100 /
                                                                totalBytes
                                                        ).toInt()

                                                    if (
                                                        percent !=
                                                        lastPercent
                                                    ) {
                                                        lastPercent =
                                                            percent

                                                        runOnUiThread {
                                                            MethodChannel(
                                                                flutterEngine
                                                                    .dartExecutor
                                                                    .binaryMessenger,
                                                                channel
                                                            ).invokeMethod(
                                                                "modelDownloadProgress",
                                                                percent
                                                            )
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                } finally {
                                    connection.disconnect()
                                }

                                if (!tempFile.renameTo(modelFile)) {
                                    throw IllegalStateException(
                                        "Unable to finalize downloaded model."
                                    )
                                }
                            }

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

                "hasCalendarPermission" -> {
                    result.success(
                        checkSelfPermission(
                            Manifest.permission.READ_CALENDAR
                        ) == PackageManager.PERMISSION_GRANTED
                    )
                }

                "requestCalendarPermission" -> {
                    requestCalendarPermission(result)
                }

                "getCalendarEvents" -> {

                    if (
                        checkSelfPermission(
                            Manifest.permission.READ_CALENDAR
                        ) != PackageManager.PERMISSION_GRANTED
                    ) {
                        result.error(
                            "CALENDAR_PERMISSION_DENIED",
                            "Calendar permission is required.",
                            null
                        )

                        return@setMethodCallHandler
                    }

                    val fromMillis =
                        call.argument<Number>(
                            "fromMillis"
                        )?.toLong()
                            ?: System.currentTimeMillis()

                    val defaultWindow =
                        90L * 24L * 60L * 60L * 1000L

                    val toMillis =
                        call.argument<Number>(
                            "toMillis"
                        )?.toLong()
                            ?: (fromMillis + defaultWindow)

                    Thread {
                        try {
                            val events =
                                readCalendarEvents(
                                    fromMillis,
                                    toMillis
                                )

                            runOnUiThread {
                                result.success(events)
                            }
                        } catch (e: Exception) {
                            runOnUiThread {
                                result.error(
                                    "CALENDAR_READ_ERROR",
                                    e.message,
                                    null
                                )
                            }
                        }
                    }.start()
                }

                else -> result.notImplemented()
            }
        }
    }

    private fun requestCalendarPermission(
        result: MethodChannel.Result
    ) {
        if (
            checkSelfPermission(
                Manifest.permission.READ_CALENDAR
            ) == PackageManager.PERMISSION_GRANTED
        ) {
            result.success(true)
            return
        }

        if (pendingCalendarPermissionResult != null) {
            result.error(
                "CALENDAR_PERMISSION_PENDING",
                "Calendar permission request already active.",
                null
            )
            return
        }

        pendingCalendarPermissionResult = result

        requestPermissions(
            arrayOf(
                Manifest.permission.READ_CALENDAR
            ),
            calendarPermissionRequestCode
        )
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(
            requestCode,
            permissions,
            grantResults
        )

        if (
            requestCode ==
            calendarPermissionRequestCode
        ) {
            val granted =
                grantResults.isNotEmpty() &&
                grantResults[0] ==
                PackageManager.PERMISSION_GRANTED

            pendingCalendarPermissionResult
                ?.success(granted)

            pendingCalendarPermissionResult = null
        }
    }

    private fun readCalendarEvents(
        fromMillis: Long,
        toMillis: Long
    ): List<Map<String, Any>> {

        val uriBuilder =
            CalendarContract.Instances
                .CONTENT_URI
                .buildUpon()

        ContentUris.appendId(
            uriBuilder,
            fromMillis
        )

        ContentUris.appendId(
            uriBuilder,
            toMillis
        )

        val projection = arrayOf(
            CalendarContract.Instances.EVENT_ID,
            CalendarContract.Instances.CALENDAR_ID,
            CalendarContract.Instances.TITLE,
            CalendarContract.Instances.BEGIN,
            CalendarContract.Instances.END,
            CalendarContract.Instances.ALL_DAY
        )

        val events =
            mutableListOf<Map<String, Any>>()

        val calendarNames =
            readCalendarNames()

        contentResolver.query(
            uriBuilder.build(),
            projection,
            null,
            null,
            "${CalendarContract.Instances.BEGIN} ASC"
        )?.use { cursor ->

            val idIndex =
                cursor.getColumnIndexOrThrow(
                    CalendarContract.Instances.EVENT_ID
                )

            val calendarIdIndex =
                cursor.getColumnIndexOrThrow(
                    CalendarContract.Instances.CALENDAR_ID
                )

            val titleIndex =
                cursor.getColumnIndexOrThrow(
                    CalendarContract.Instances.TITLE
                )

            val beginIndex =
                cursor.getColumnIndexOrThrow(
                    CalendarContract.Instances.BEGIN
                )

            val endIndex =
                cursor.getColumnIndexOrThrow(
                    CalendarContract.Instances.END
                )

            val allDayIndex =
                cursor.getColumnIndexOrThrow(
                    CalendarContract.Instances.ALL_DAY
                )

            while (cursor.moveToNext()) {

                val eventId =
                    cursor.getLong(idIndex)

                val calendarId =
                    cursor.getLong(calendarIdIndex)

                val title =
                    cursor.getString(titleIndex)
                        ?: "Untitled event"

                val begin =
                    cursor.getLong(beginIndex)

                val end =
                    cursor.getLong(endIndex)

                val allDay =
                    cursor.getInt(allDayIndex) != 0

                events.add(
                    mapOf(
                        "id" to eventId.toString(),
                        "calendarName" to
                            (calendarNames[calendarId] ?: ""),
                        "title" to title,
                        "startMillis" to begin,
                        "endMillis" to end,
                        "allDay" to allDay
                    )
                )
            }
        }

        return events
    }

    private fun readCalendarNames():
        Map<Long, String> {

        val names =
            mutableMapOf<Long, String>()

        val projection = arrayOf(
            CalendarContract.Calendars._ID,
            CalendarContract.Calendars.CALENDAR_DISPLAY_NAME
        )

        contentResolver.query(
            CalendarContract.Calendars.CONTENT_URI,
            projection,
            null,
            null,
            null
        )?.use { cursor ->

            val idIndex =
                cursor.getColumnIndexOrThrow(
                    CalendarContract.Calendars._ID
                )

            val nameIndex =
                cursor.getColumnIndexOrThrow(
                    CalendarContract.Calendars.CALENDAR_DISPLAY_NAME
                )

            while (cursor.moveToNext()) {
                names[cursor.getLong(idIndex)] =
                    cursor.getString(nameIndex) ?: ""
            }
        }

        return names
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




