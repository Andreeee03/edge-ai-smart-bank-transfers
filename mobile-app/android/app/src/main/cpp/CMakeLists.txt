cmake_minimum_required(VERSION 3.31.6)

project(edge_ai_native)

add_library(
        edge_ai_native
        SHARED
        native_bridge.cpp
)