cmake_minimum_required(VERSION 3.31.6)

project(edge_ai_native)

set(LLAMA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../llama.cpp")

set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)

set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL ON CACHE BOOL "" FORCE)
set(GGML_CPU_ALL_VARIANTS ON CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE OFF CACHE BOOL "" FORCE)

add_subdirectory(
        "${LLAMA_DIR}"
        "${CMAKE_CURRENT_BINARY_DIR}/llama"
)

add_library(
        edge_ai_native
        SHARED
        native_bridge.cpp
)

target_include_directories(
        edge_ai_native
        PRIVATE
        "${LLAMA_DIR}/include"
)

target_link_libraries(
        edge_ai_native
        llama
        log
)
