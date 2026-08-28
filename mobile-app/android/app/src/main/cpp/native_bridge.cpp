#include <jni.h>
#include <string>
#include <mutex>
#include <vector>
#include <sstream>
#include <android/log.h>

#include "llama.h"
#include "ggml-backend.h"

static llama_model * g_model = nullptr;
static llama_context * g_ctx = nullptr;

static bool g_backend_initialized = false;
static bool g_prompt_decoded = false;

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
        jobject,
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
            "Native init FAILED"
        );
    }

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
        jobject) {

    std::lock_guard<std::mutex> lock(g_mutex);

    std::string result =
        "llama.cpp bridge OK\n";

    const char * info =
        llama_print_system_info();

    if (info != nullptr) {
        result += info;
    }

    return env->NewStringUTF(
        result.c_str()
    );
}


extern "C"
JNIEXPORT jstring JNICALL
Java_com_example_edge_1ai_1smart_1bank_1transfers_MainActivity_nativeLoadModel(
        JNIEnv * env,
        jobject,
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
            "Model load FAILED: invalid path"
        );
    }

    llama_model_params params =
        llama_model_default_params();

    g_model =
        llama_model_load_from_file(
            path,
            params
        );

    env->ReleaseStringUTFChars(
        modelPath,
        path
    );

    if (g_model == nullptr) {
        return env->NewStringUTF(
            "Model load FAILED"
        );
    }

    return env->NewStringUTF(
        "Model load OK"
    );
}


extern "C"
JNIEXPORT jstring JNICALL
Java_com_example_edge_1ai_1smart_1bank_1transfers_MainActivity_nativeCreateContext(
        JNIEnv * env,
        jobject) {

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

    llama_context_params params =
        llama_context_default_params();

    params.n_ctx = 512;
    params.n_batch = 512;
    params.n_threads = 4;
    params.n_threads_batch = 4;

    g_ctx =
        llama_init_from_model(
            g_model,
            params
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


static int32_t tokenize_text(
        const llama_vocab * vocab,
        const std::string & text,
        std::vector<llama_token> & tokens) {

    tokens.resize(
        text.size() + 16
    );

    int32_t n_tokens =
        llama_tokenize(
            vocab,
            text.c_str(),
            static_cast<int32_t>(text.size()),
            tokens.data(),
            static_cast<int32_t>(tokens.size()),
            true,
            false
        );

    if (n_tokens < 0) {

        tokens.resize(
            -n_tokens
        );

        n_tokens =
            llama_tokenize(
                vocab,
                text.c_str(),
                static_cast<int32_t>(text.size()),
                tokens.data(),
                static_cast<int32_t>(tokens.size()),
                true,
                false
            );
    }

    if (n_tokens >= 0) {
        tokens.resize(n_tokens);
    }

    return n_tokens;
}


extern "C"
JNIEXPORT jstring JNICALL
Java_com_example_edge_1ai_1smart_1bank_1transfers_MainActivity_nativeTokenize(
        JNIEnv * env,
        jobject,
        jstring prompt) {

    std::lock_guard<std::mutex> lock(g_mutex);

    if (g_model == nullptr) {
        return env->NewStringUTF(
            "Tokenization FAILED: model not loaded"
        );
    }

    const llama_vocab * vocab =
        llama_model_get_vocab(g_model);

    const char * chars =
        env->GetStringUTFChars(
            prompt,
            nullptr
        );

    if (chars == nullptr) {
        return env->NewStringUTF(
            "Tokenization FAILED"
        );
    }

    std::string text(chars);

    env->ReleaseStringUTFChars(
        prompt,
        chars
    );

    std::vector<llama_token> tokens;

    const int32_t n_tokens =
        tokenize_text(
            vocab,
            text,
            tokens
        );

    if (n_tokens < 0) {
        return env->NewStringUTF(
            "Tokenization FAILED"
        );
    }

    std::ostringstream out;

    out << "Tokenization OK - tokens="
        << n_tokens
        << "\nToken IDs: ";

    const size_t max_show = 40;

    for (
        size_t i = 0;
        i < tokens.size() && i < max_show;
        ++i
    ) {
        out << tokens[i];

        if (
            i + 1 < tokens.size() &&
            i + 1 < max_show
        ) {
            out << ", ";
        }
    }

    if (tokens.size() > max_show) {
        out << ", ...";
    }

    return env->NewStringUTF(
        out.str().c_str()
    );
}


extern "C"
JNIEXPORT jstring JNICALL
Java_com_example_edge_1ai_1smart_1bank_1transfers_MainActivity_nativeDecode(
        JNIEnv * env,
        jobject,
        jstring prompt) {

    std::lock_guard<std::mutex> lock(g_mutex);

    if (g_model == nullptr) {
        return env->NewStringUTF(
            "Decode FAILED: model not loaded"
        );
    }

    if (g_ctx == nullptr) {
        return env->NewStringUTF(
            "Decode FAILED: context not created"
        );
    }

    if (g_prompt_decoded) {
        return env->NewStringUTF(
            "Decode already completed OK"
        );
    }

    const llama_vocab * vocab =
        llama_model_get_vocab(g_model);

    const char * chars =
        env->GetStringUTFChars(
            prompt,
            nullptr
        );

    if (chars == nullptr) {
        return env->NewStringUTF(
            "Decode FAILED: invalid prompt"
        );
    }

    std::string text(chars);

    env->ReleaseStringUTFChars(
        prompt,
        chars
    );

    std::vector<llama_token> tokens;

    const int32_t n_tokens =
        tokenize_text(
            vocab,
            text,
            tokens
        );

    if (n_tokens <= 0) {
        return env->NewStringUTF(
            "Decode FAILED: tokenization error"
        );
    }

    if (n_tokens > 512) {
        return env->NewStringUTF(
            "Decode FAILED: prompt exceeds n_ctx=512"
        );
    }

    llama_batch batch =
        llama_batch_get_one(
            tokens.data(),
            n_tokens
        );

    const int32_t result =
        llama_decode(
            g_ctx,
            batch
        );

    if (result != 0) {

        std::ostringstream error;

        error
            << "Decode FAILED - code="
            << result;

        return env->NewStringUTF(
            error.str().c_str()
        );
    }

    llama_synchronize(g_ctx);

    float * logits =
        llama_get_logits(g_ctx);

    if (logits == nullptr) {
        return env->NewStringUTF(
            "Decode FAILED: logits unavailable"
        );
    }

    g_prompt_decoded = true;

    std::ostringstream out;

    out
        << "Decode OK"
        << "\nPrompt tokens: "
        << n_tokens
        << "\nLogits ready";

    return env->NewStringUTF(
        out.str().c_str()
    );
}


extern "C"
JNIEXPORT jstring JNICALL
Java_com_example_edge_1ai_1smart_1bank_1transfers_MainActivity_nativeSampleFirstToken(
        JNIEnv * env,
        jobject) {

    std::lock_guard<std::mutex> lock(g_mutex);

    if (g_model == nullptr) {
        return env->NewStringUTF(
            "Sampling FAILED: model not loaded"
        );
    }

    if (g_ctx == nullptr) {
        return env->NewStringUTF(
            "Sampling FAILED: context not created"
        );
    }

    if (!g_prompt_decoded) {
        return env->NewStringUTF(
            "Sampling FAILED: decode prompt first"
        );
    }

    const llama_vocab * vocab =
        llama_model_get_vocab(g_model);

    llama_sampler * sampler =
        llama_sampler_init_greedy();

    if (sampler == nullptr) {
        return env->NewStringUTF(
            "Sampling FAILED: sampler unavailable"
        );
    }

    const llama_token token =
        llama_sampler_sample(
            sampler,
            g_ctx,
            -1
        );

    llama_sampler_free(sampler);

    const bool is_eog =
        llama_vocab_is_eog(
            vocab,
            token
        );

    char buffer[256];

    int32_t piece_length =
        llama_token_to_piece(
            vocab,
            token,
            buffer,
            sizeof(buffer),
            0,
            false
        );

    std::string piece;

    if (piece_length < 0) {

        std::vector<char> large_buffer(
            static_cast<size_t>(-piece_length)
        );

        piece_length =
            llama_token_to_piece(
                vocab,
                token,
                large_buffer.data(),
                static_cast<int32_t>(
                    large_buffer.size()
                ),
                0,
                false
            );

        if (piece_length < 0) {
            return env->NewStringUTF(
                "Sampling FAILED: token conversion error"
            );
        }

        piece.assign(
            large_buffer.data(),
            piece_length
        );

    } else {

        piece.assign(
            buffer,
            piece_length
        );
    }

    std::ostringstream out;

    out
        << "Greedy sample OK"
        << "\nToken ID: "
        << token
        << "\nPiece: "
        << piece
        << "\nEOG: "
        << (is_eog ? "yes" : "no");

    return env->NewStringUTF(
        out.str().c_str()
    );
}


static bool token_to_text(
        const llama_vocab * vocab,
        llama_token token,
        std::string & piece) {

    char buffer[256];

    int32_t length =
        llama_token_to_piece(
            vocab,
            token,
            buffer,
            sizeof(buffer),
            0,
            false
        );

    if (length >= 0) {

        piece.assign(
            buffer,
            length
        );

        return true;
    }

    std::vector<char> large_buffer(
        static_cast<size_t>(-length)
    );

    length =
        llama_token_to_piece(
            vocab,
            token,
            large_buffer.data(),
            static_cast<int32_t>(
                large_buffer.size()
            ),
            0,
            false
        );

    if (length < 0) {
        return false;
    }

    piece.assign(
        large_buffer.data(),
        length
    );

    return true;
}


extern "C"
JNIEXPORT jstring JNICALL
Java_com_example_edge_1ai_1smart_1bank_1transfers_MainActivity_nativeGenerate(
        JNIEnv * env,
        jobject,
        jstring prompt) {

    std::lock_guard<std::mutex> lock(g_mutex);

    if (g_model == nullptr) {
        return env->NewStringUTF(
            "Generation FAILED: model not loaded"
        );
    }

    if (g_ctx == nullptr) {
        return env->NewStringUTF(
            "Generation FAILED: context not created"
        );
    }

    const char * chars =
        env->GetStringUTFChars(
            prompt,
            nullptr
        );

    if (chars == nullptr) {
        return env->NewStringUTF(
            "Generation FAILED: invalid prompt"
        );
    }

    std::string text(chars);

    env->ReleaseStringUTFChars(
        prompt,
        chars
    );

    const llama_vocab * vocab =
        llama_model_get_vocab(g_model);

    if (vocab == nullptr) {
        return env->NewStringUTF(
            "Generation FAILED: vocab unavailable"
        );
    }

    // Ogni nuova richiesta parte con KV cache pulita.
    llama_memory_t memory =
        llama_get_memory(g_ctx);

    llama_memory_clear(
        memory,
        true
    );

    g_prompt_decoded = false;

    std::vector<llama_token> prompt_tokens;

    const int32_t n_prompt_tokens =
        tokenize_text(
            vocab,
            text,
            prompt_tokens
        );

    if (n_prompt_tokens <= 0) {
        return env->NewStringUTF(
            "Generation FAILED: tokenization error"
        );
    }

    constexpr int32_t N_CTX = 512;
    constexpr int32_t MAX_NEW_TOKENS = 100;

    if (n_prompt_tokens >= N_CTX) {
        return env->NewStringUTF(
            "Generation FAILED: prompt too long"
        );
    }

    llama_batch prompt_batch =
        llama_batch_get_one(
            prompt_tokens.data(),
            n_prompt_tokens
        );

    const int32_t prompt_decode_result =
        llama_decode(
            g_ctx,
            prompt_batch
        );

    if (prompt_decode_result != 0) {

        std::ostringstream error;

        error
            << "Generation FAILED: prompt decode code="
            << prompt_decode_result;

        return env->NewStringUTF(
            error.str().c_str()
        );
    }

    llama_sampler * sampler =
        llama_sampler_init_greedy();

    if (sampler == nullptr) {
        return env->NewStringUTF(
            "Generation FAILED: sampler unavailable"
        );
    }

    const int32_t available_tokens =
        N_CTX - n_prompt_tokens;

    const int32_t generation_limit =
        available_tokens < MAX_NEW_TOKENS
            ? available_tokens
            : MAX_NEW_TOKENS;

    std::string output;

    int32_t generated_tokens = 0;
    bool reached_eog = false;

    for (
        int32_t i = 0;
        i < generation_limit;
        ++i
    ) {

        const llama_token token =
            llama_sampler_sample(
                sampler,
                g_ctx,
                -1
            );

        if (
            llama_vocab_is_eog(
                vocab,
                token
            )
        ) {
            reached_eog = true;
            break;
        }

        std::string piece;

        if (
            !token_to_text(
                vocab,
                token,
                piece
            )
        ) {
            llama_sampler_free(sampler);

            return env->NewStringUTF(
                "Generation FAILED: token conversion error"
            );
        }

        output += piece;

        generated_tokens++;

        llama_token next_token = token;

        llama_batch token_batch =
            llama_batch_get_one(
                &next_token,
                1
            );

        const int32_t decode_result =
            llama_decode(
                g_ctx,
                token_batch
            );

        if (decode_result != 0) {

            llama_sampler_free(sampler);

            std::ostringstream error;

            error
                << "Generation FAILED: token decode code="
                << decode_result;

            return env->NewStringUTF(
                error.str().c_str()
            );
        }
    }

    llama_sampler_free(sampler);

    std::ostringstream result;

    result
        << "Generation OK"
        << "\nPrompt tokens: "
        << n_prompt_tokens
        << "\nGenerated tokens: "
        << generated_tokens
        << "\nEOG: "
        << (reached_eog ? "yes" : "no")
        << "\n\nOutput:\n"
        << output;

    return env->NewStringUTF(
        result.str().c_str()
    );
}
