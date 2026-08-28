#include <jni.h>

extern "C"
JNIEXPORT jstring JNICALL
Java_com_example_edge_1ai_1smart_1bank_1transfers_MainActivity_nativePing(
        JNIEnv* env,
        jobject /* this */) {

    return env->NewStringUTF("C++ JNI bridge OK");
}