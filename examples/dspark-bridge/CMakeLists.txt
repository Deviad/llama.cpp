set(TARGET llama-dspark-bridge)
add_executable(${TARGET} dspark-bridge.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
# Story S31 AC4: identical source, separate output binary. Lets us rebuild
# with the new `dspark_cycle_split` op while an older bridge might still be
# running (avoids macOS ETXTBSY on the in-use executable).
set(TARGET_SPLIT llama-dspark-bridge-split)
add_executable(${TARGET_SPLIT} dspark-bridge.cpp)
install(TARGETS ${TARGET_SPLIT} RUNTIME)
target_link_libraries(${TARGET_SPLIT} PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET_SPLIT} PRIVATE cxx_std_17)
# Story S32: C++ fused DFlash runtime. Same source, separate binary so the
# new dflash_load/dflash_forward ops don't disrupt any in-flight capture
# using llama-dspark-bridge-ml (avoids ETXTBSY on the live executable).
set(TARGET_DFLASH llama-dspark-bridge-dflash)
add_executable(${TARGET_DFLASH} dspark-bridge.cpp)
install(TARGETS ${TARGET_DFLASH} RUNTIME)
target_link_libraries(${TARGET_DFLASH} PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET_DFLASH} PRIVATE cxx_std_17)
# Story S30: parallel matmuls in the trained C++ drafter. Link Apple's
# Accelerate framework (cblas_sgemm) for the big V-dim matmuls — this is
# already a transitive dependency via ggml-blas, but we link it explicitly
# here so the bridge can call cblas_sgemm directly. On non-Apple platforms
# fall back to OpenMP if available.
if(APPLE)
    find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
    target_link_libraries(${TARGET} PRIVATE ${ACCELERATE_FRAMEWORK})
    target_compile_definitions(${TARGET} PRIVATE USE_ACCELERATE=1)
    target_link_libraries(${TARGET_SPLIT} PRIVATE ${ACCELERATE_FRAMEWORK})
    target_compile_definitions(${TARGET_SPLIT} PRIVATE USE_ACCELERATE=1)
    target_link_libraries(${TARGET_DFLASH} PRIVATE ${ACCELERATE_FRAMEWORK})
    target_compile_definitions(${TARGET_DFLASH} PRIVATE USE_ACCELERATE=1)
else()
    find_package(OpenMP QUIET)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(${TARGET} PRIVATE OpenMP::OpenMP_CXX)
        target_link_libraries(${TARGET_SPLIT} PRIVATE OpenMP::OpenMP_CXX)
        target_link_libraries(${TARGET_DFLASH} PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()
