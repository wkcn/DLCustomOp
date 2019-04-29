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

typedef void* CustomOpHandle;

typedef struct CCustomOp{
  void (*forward)(CustomOpHandle self, CArgs* args, CArgs* tensors);
  void (*backward)(CustomOpHandle self, CArgs* args, CArgs* tensors);
  void (*input_names)(CustomOpHandle self, CArgs* names);
  void (*output_names)(CustomOpHandle self, CArgs* names);
  void (*infer_shape)(CustomOpHandle self, CArgs* inshapes, CArgs* outshapes);

  CustomOpHandle (*creator)();
  void (*deleter)(CustomOpHandle self);
} CCustomOp; 

#ifdef __cplusplus
}
#endif

#endif  // CUSTOM_OP_H_
