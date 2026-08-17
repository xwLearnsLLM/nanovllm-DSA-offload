#ifndef NANOVLLM_DSA_A5_OPS_LOG_COMPAT_H
#define NANOVLLM_DSA_A5_OPS_LOG_COMPAT_H

#include <cstdio>

// The upstream vLLM-Ascend tiler uses logging helpers from its monorepo build.
// Keep this standalone operator project independent from those private headers.
#ifndef OP_LOGI
#define OP_LOGI(op_name, ...) \
    do {                       \
        (void)(op_name);       \
    } while (0)
#endif

#ifndef OP_LOGW
#define OP_LOGW(op_name, ...)          \
    do {                               \
        (void)(op_name);               \
        std::fprintf(stderr, "[WARN] "); \
        std::fprintf(stderr, __VA_ARGS__); \
        std::fprintf(stderr, "\n");    \
    } while (0)
#endif

#ifndef OP_LOGE
#define OP_LOGE(op_name, ...)           \
    do {                                \
        (void)(op_name);                \
        std::fprintf(stderr, "[ERROR] "); \
        std::fprintf(stderr, __VA_ARGS__); \
        std::fprintf(stderr, "\n");     \
    } while (0)
#endif

#ifndef OPS_REPORT_VECTOR_INNER_ERR
#define OPS_REPORT_VECTOR_INNER_ERR(op_name, ...) \
    OP_LOGE(op_name, __VA_ARGS__)
#endif

#ifndef OP_CHECK_IF
#define OP_CHECK_IF(condition, log_statement, action) \
    do {                                               \
        if (condition) {                               \
            log_statement;                             \
            action;                                    \
        }                                              \
    } while (0)
#endif

#ifndef OP_CHECK_NULL_WITH_CONTEXT
#define OP_CHECK_NULL_WITH_CONTEXT(context, value)                  \
    do {                                                            \
        (void)(context);                                            \
        if ((value) == nullptr) {                                   \
            OP_LOGE("standalone_op", "%s is nullptr.", #value); \
            return ge::GRAPH_FAILED;                                \
        }                                                           \
    } while (0)
#endif

// CANN releases do not consistently install err/ops_err.h. These helpers
// only report validation failures; callers still return the graph status.
#ifndef NANOVLLM_OP_LOG_INVALID
#define NANOVLLM_OP_LOG_INVALID(op_name, ...)           \
    do {                                                \
        (void)(op_name);                                \
        std::fprintf(stderr, "[ERROR] invalid operator argument\n"); \
    } while (0)
#endif

#ifndef OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON
#define OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(op_name, ...) \
    NANOVLLM_OP_LOG_INVALID(op_name, __VA_ARGS__)
#endif

#ifndef OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON
#define OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(op_name, ...) \
    NANOVLLM_OP_LOG_INVALID(op_name, __VA_ARGS__)
#endif

#ifndef OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON
#define OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(op_name, ...) \
    NANOVLLM_OP_LOG_INVALID(op_name, __VA_ARGS__)
#endif

#ifndef OP_LOGE_FOR_INVALID_SHAPE
#define OP_LOGE_FOR_INVALID_SHAPE(op_name, ...) \
    NANOVLLM_OP_LOG_INVALID(op_name, __VA_ARGS__)
#endif

#ifndef OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON
#define OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(op_name, ...) \
    NANOVLLM_OP_LOG_INVALID(op_name, __VA_ARGS__)
#endif

#ifndef OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON
#define OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(op_name, ...) \
    NANOVLLM_OP_LOG_INVALID(op_name, __VA_ARGS__)
#endif

#ifndef OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON
#define OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(op_name, ...) \
    NANOVLLM_OP_LOG_INVALID(op_name, __VA_ARGS__)
#endif

#ifndef OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON
#define OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(op_name, ...) \
    NANOVLLM_OP_LOG_INVALID(op_name, __VA_ARGS__)
#endif

#ifndef OP_LOGE_FOR_INVALID_VALUE
#define OP_LOGE_FOR_INVALID_VALUE(op_name, ...) \
    NANOVLLM_OP_LOG_INVALID(op_name, __VA_ARGS__)
#endif

#ifndef OP_LOGE_FOR_INVALID_VALUES_WITH_REASON
#define OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(op_name, ...) \
    NANOVLLM_OP_LOG_INVALID(op_name, __VA_ARGS__)
#endif

#ifndef OP_LOGE_FOR_INVALID_VALUE_WITH_REASON
#define OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(op_name, ...) \
    NANOVLLM_OP_LOG_INVALID(op_name, __VA_ARGS__)
#endif

// Some reference kernels use the older OPS_* spellings from
// error/ops_error.h.  That private header is not shipped by every CANN 9.1
// Ascend 950 package, so keep the small subset needed by this standalone
// project local.
#ifndef OPS_LOG_E
#define OPS_LOG_E(op_name, ...) OP_LOGE(op_name, __VA_ARGS__)
#endif

#ifndef OPS_ERR_IF
#define OPS_ERR_IF(condition, log_statement, action) \
    OP_CHECK_IF(condition, log_statement, action)
#endif

#ifndef OPS_LOG_E_IF_NULL
#define OPS_LOG_E_IF_NULL(context, value, action)              \
    do {                                                       \
        if ((value) == nullptr) {                              \
            OP_LOGE(context, "%s is nullptr.", #value);      \
            action;                                            \
        }                                                      \
    } while (0)
#endif

#endif
