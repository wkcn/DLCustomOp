#ifndef CUSTOM_OP_H_
#define CUSTOM_OP_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  void* values;
  int* type_codes;
  int num_args;
} CArgs;

typedef struct CCustomOp{
  void (*forward)(CArgs* args, CArgs* tensors);
  void (*backward)(CArgs* args, CArgs* tensors);
  void (*input_names)(CArgs** names);
  void (*output_names)(CArgs** names);
  void (*infer_shape)(CArgs *inputs, CArgs **outputs);

  void (*init)(struct CCustomOp* self);
  void (*deleter)(struct CCustomOp* self);
  void* manager_ctx;
} CCustomOp; 

#ifdef __cplusplus
}
#endif

#endif  // CUSTOM_OP_H_
