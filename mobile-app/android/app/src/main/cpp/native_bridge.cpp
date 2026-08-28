#include <jni.h>
#include <string>

#include "llama.h"

extern "C"
JNIEXPORT jstring JNICALL
Java_com_example_edge_1ai_1smart_1bank_1transfers_MainActivity_nativePing(
        JNIEnv* env,
        jobject /* this */) {

    llama_backend_init();

    const char* systemInfo = llama_print_system_info();

    std::string result = "llama.cpp bridge OK\n";

    if (systemInfo != nullptr) {
        result += systemInfo;
    }

    return env->NewStringUTF(result.c_str());
}