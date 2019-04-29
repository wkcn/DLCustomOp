#ifndef CUSTOM_OP_HPP_
#define CUSTOM_OP_HPP_

#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <initializer_list>
#include "c_custom_op.h"
using namespace std;

class DLTensor {
public:
  DLTensor& operator+(float val) {
    std::cout << "DLTensor += " << val << std::endl;
    return *this;
  }
};

class TArgsManager;

// it's similar to std::any
class TArg {
public:
  template <typename T>
  T& get(T default_value) {
    // TODO
    return *(new T(default_value));
  }
  template <typename T>
  T& get() {
    // TODO
    return *(new T());
  }
private:
  void* ptr_;
};

class TArgs {
public:
  TArgs(CArgs* cargs, TArgsManager* allocator) : cargs_(cargs), allocator_(allocator) {
  }
  TArg operator[](const size_t i) {
    return TArg();
  }
  TArg operator[](const string s) {
    return TArg();
  }
  TArgs& operator=(const std::initializer_list<int>& lst) {
    return *this;
  }
  TArgs& operator=(const std::vector<std::string>& vs) {
    // We will use TArgsManager to allocate and free the memory
    return *this;
  }
private:
  CArgs* cargs_;
  TArgsManager* allocator_;
};

// Use TArgsManager to manager the memory
class TArgsManager {
public:
  TArgsManager() {
  }
  void Alloc(size_t) {
  }
  void Free(void*) {
  }
  ~TArgsManager() {
    // Free memory
  }
  TArgs operator()(CArgs* cargs) {
    return TArgs(cargs, this);
  }
};

class CustomOp {
public:
  CustomOp() {
  }

  template <typename op_type>
  static CCustomOp Register() {
    CCustomOp c_op;
    // fill the attributes of CCustomOp
    c_op.forward = [](CustomOpHandle pself, CArgs* args, CArgs* tensors) {
      CustomOp* self = static_cast<CustomOp*>(pself);
      TArgsManager& ToTArgs = self->ToTArgs;
      self->Forward(ToTArgs(args), ToTArgs(tensors));
    };
    c_op.backward = [](CustomOpHandle pself, CArgs* args, CArgs* tensors) {
      CustomOp* self = static_cast<CustomOp*>(pself);
      TArgsManager& ToTArgs = self->ToTArgs;
      self->Backward(ToTArgs(args), ToTArgs(tensors));
    };
    c_op.input_names = [](CustomOpHandle pself, CArgs* names) {
      CustomOp* self = static_cast<CustomOp*>(pself);
      TArgsManager& ToTArgs = self->ToTArgs;
      std::vector<std::string> vs = self->InputNames();
      ToTArgs(names) = vs;
    };
    c_op.output_names = [](CustomOpHandle pself, CArgs* names) {
      CustomOp* self = static_cast<CustomOp*>(pself);
      TArgsManager& ToTArgs = self->ToTArgs;
      std::vector<std::string> vs = self->OutputNames();
      ToTArgs(names) = vs;
    };
    c_op.infer_shape = [](CustomOpHandle pself, CArgs* inshapes, CArgs* outshapes) {
      CustomOp* self = static_cast<CustomOp*>(pself);
      TArgsManager& ToTArgs = self->ToTArgs;
      self->InferShape(ToTArgs(inshapes), ToTArgs(outshapes));
    };
    c_op.init = []{return static_cast<void*>(new op_type());};
    c_op.deleter = [](CustomOpHandle pself) {
      CustomOp* self = static_cast<CustomOp*>(pself);
    }; 
    return c_op;
  }

  virtual ~CustomOp() {
  }
  virtual void Init() {};
  virtual void Deleter() {};
  virtual void InferShape(TArgs ishapes, TArgs oshapes) = 0;
  virtual void Forward(TArgs args, TArgs tensors) = 0;
  virtual void Backward(TArgs args, TArgs tensors) = 0;
  virtual std::vector<std::string> InputNames() = 0;
  virtual std::vector<std::string> OutputNames() = 0;
private:
  TArgsManager ToTArgs;
};

#endif
