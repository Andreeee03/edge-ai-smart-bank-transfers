#include <jni.h>
#include <string>
#include <mutex>
#include <android/log.h>

#include "llama.h"
#include "ggml-backend.h"

static llama_model * g_model = nullptr;
static llama_context * g_ctx = nullptr;
static bool g_backend_initialized = false;
static std::mutex g_mutex;

static void android_log_callback(
        ggml_log_level level,
        const char * text,
        void * /* user_data */) {

    int priority = ANDROID_LOG_INFO;

    if (level == GGML_LOG_LEVEL_ERROR) {
        priority = ANDROID_LOG_ERROR;
    } else if (level == GGML_LOG_LEVEL_WARN) {
        priority = ANDROID_LOG_WARN;
    } else if (level == GGML_LOG_LEVEL_DEBUG) {
        priority = ANDROID_LOG_DEBUG;
    }

    __android_log_print(
        priority,
        "EdgeAI-Llama",
        "%s",
        text
    );
}


extern "C"
JNIEXPORT jstring JNICALL
Java_com_example_edge_1ai_1smart_1bank_1transfers_MainActivity_nativeInit(
        JNIEnv * env,
        jobject /* this */,
        jstring nativeLibDir) {

    std::lock_guard<std::mutex> lock(g_mutex);

    if (g_backend_initialized) {
        return env->NewStringUTF(
            "Native backend already initialized"
        );
    }

    llama_log_set(
        android_log_callback,
        nullptr
    );

    const char * native_lib_dir =
        env->GetStringUTFChars(
            nativeLibDir,
            nullptr
        );

    if (native_lib_dir == nullptr) {
        return env->NewStringUTF(
            "Native init FAILED: invalid native library directory"
        );
    }

    __android_log_print(
        ANDROID_LOG_INFO,
        "EdgeAI-Llama",
        "Loading GGML backends from: %s",
        native_lib_dir
    );

    ggml_backend_load_all_from_path(
        native_lib_dir
    );

    env->ReleaseStringUTFChars(
        nativeLibDir,
        native_lib_dir
    );

    llama_backend_init();

    g_backend_initialized = true;

    return env->NewStringUTF(
        "Native backend initialized OK"
    );
}


extern "C"
JNIEXPORT jstring JNICALL
Java_com_example_edge_1ai_1smart_1bank_1transfers_MainActivity_nativePing(
        JNIEnv * env,
        jobject /* this */) {

    std::lock_guard<std::mutex> lock(g_mutex);

    std::string result =
        "llama.cpp bridge OK\n";

    const char * system_info =
        llama_print_system_info();

    if (system_info != nullptr) {
        result += system_info;
    }

    return env->NewStringUTF(
        result.c_str()
    );
}


extern "C"
JNIEXPORT jstring JNICALL
Java_com_example_edge_1ai_1smart_1bank_1transfers_MainActivity_nativeLoadModel(
        JNIEnv * env,
        jobject /* this */,
        jstring modelPath) {

    std::lock_guard<std::mutex> lock(g_mutex);

    if (!g_backend_initialized) {
        return env->NewStringUTF(
            "Model load FAILED: backend not initialized"
        );
    }

    if (g_model != nullptr) {
        return env->NewStringUTF(
            "Model already loaded OK"
        );
    }

    const char * path =
        env->GetStringUTFChars(
            modelPath,
            nullptr
        );

    if (path == nullptr) {
        return env->NewStringUTF(
            "Model load FAILED: invalid model path"
        );
    }

    __android_log_print(
        ANDROID_LOG_INFO,
        "EdgeAI-Llama",
        "Loading model: %s",
        path
    );

    llama_model_params model_params =
        llama_model_default_params();

    g_model =
        llama_model_load_from_file(
            path,
            model_params
        );

    env->ReleaseStringUTFChars(
        modelPath,
        path
    );

    if (g_model == nullptr) {
        __android_log_print(
            ANDROID_LOG_ERROR,
            "EdgeAI-Llama",
            "llama_model_load_from_file returned nullptr"
        );

        return env->NewStringUTF(
            "Model load FAILED"
        );
    }

    __android_log_print(
        ANDROID_LOG_INFO,
        "EdgeAI-Llama",
        "Model loaded successfully"
    );

    return env->NewStringUTF(
        "Model load OK"
    );
}


extern "C"
JNIEXPORT jstring JNICALL
Java_com_example_edge_1ai_1smart_1bank_1transfers_MainActivity_nativeCreateContext(
        JNIEnv * env,
        jobject /* this */) {

    std::lock_guard<std::mutex> lock(g_mutex);

    if (g_model == nullptr) {
        return env->NewStringUTF(
            "Context creation FAILED: model not loaded"
        );
    }

    if (g_ctx != nullptr) {
        return env->NewStringUTF(
            "Context already created OK"
        );
    }

    llama_context_params ctx_params =
        llama_context_default_params();

    ctx_params.n_ctx = 512;
    ctx_params.n_batch = 512;
    ctx_params.n_threads = 4;
    ctx_params.n_threads_batch = 4;

    g_ctx = llama_init_from_model(
        g_model,
        ctx_params
    );

    if (g_ctx == nullptr) {
        return env->NewStringUTF(
            "Context creation FAILED"
        );
    }

    return env->NewStringUTF(
        "Context creation OK - n_ctx=512"
    );
}
