# 下面的代码定义了一个名为 custom_plugins 的共享库目标。add_library 命令会在构建过程中生成一个动态链接库，使得插件可以在 TensorRT 推理时被加载并使用。通过 SHARED 关键字，CMake 知道要将该目标编译为共享库而不是静态库或可执行文件。
# 定义目标 custom_plugins
add_library(custom_plugins SHARED)

# 添加头文件搜索路径
target_include_directories(custom_plugins PRIVATE
    ${PROJECT_SOURCE_DIR}/plugin
)

# 使用file(GLOB_RECURSE ...)递归搜索指定目录下的所有.cpp和.cu文件
file(GLOB_RECURSE SOURCES
    ${PROJECT_SOURCE_DIR}/plugin/common/*.cpp
    ${PROJECT_SOURCE_DIR}/plugin/common/*.cu
    ${PROJECT_SOURCE_DIR}/plugin/efficientRotatedNMSPlugin/*.cpp
    ${PROJECT_SOURCE_DIR}/plugin/efficientRotatedNMSPlugin/*.cu
    ${PROJECT_SOURCE_DIR}/plugin/efficientIdxNMSPlugin/*.cpp
    ${PROJECT_SOURCE_DIR}/plugin/efficientIdxNMSPlugin/*.cu
)

# 将搜索到的文件添加到目标中
target_sources(custom_plugins PRIVATE ${SOURCES})

configure_cuda_trt(custom_plugins)
set_compile_options(custom_plugins)
set_target_properties(custom_plugins PROPERTIES OUTPUT_NAME custom_plugins)
if(MSVC)
    set_target_properties(custom_plugins PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE ${PROJECT_SOURCE_DIR}/lib/plugin) # 设置Release模式下输出目录
    set_target_properties(custom_plugins PROPERTIES ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${PROJECT_SOURCE_DIR}/lib/plugin)
    set_target_properties(custom_plugins PROPERTIES RUNTIME_OUTPUT_DIRECTORY_DEBUG ${PROJECT_SOURCE_DIR}/lib/plugin)   # 设置Debug模式下输出目录
    set_target_properties(custom_plugins PROPERTIES ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${PROJECT_SOURCE_DIR}/lib/plugin)
else()
    set_target_properties(custom_plugins PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib/plugin)
endif()