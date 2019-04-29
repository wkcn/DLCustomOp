#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <initializer_list>
#include "custom_op.h"
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

class AddScalarOp : public CustomOp {
public:
  AddScalarOp() {
    std::cout << "Init AddScalarOp" << std::endl;
  }
  ~AddScalarOp() {
    std::cout << "Delete AddScalarOp" << std::endl;
  }
  void Forward(TArgs args, TArgs tensors) override {
    float scalar = args["scalar"].get<float>(0);
    DLTensor* input = tensors[0].get<DLTensor*>();
    DLTensor* output = tensors[1].get<DLTensor*>();
    *output = *input + scalar;
    std::cout << "Forward" << std::endl;
  }
  void Backward(TArgs args, TArgs tensors) override {
    std::cout << "Backward" << std::endl;
  }
  void InferShape(TArgs ishapes, TArgs oshapes) override {
    std::cout << "InferShape" << std::endl;
    oshapes = {1};
  }
  std::vector<std::string> InputNames() override {
    return {"data"};
  }
  std::vector<std::string> OutputNames() override {
    return {"output"};
  }
private:
};

void PrintStrings(std::vector<std::string> &vs) {
  bool first = true;
  for (std::string &s : vs) {
    if (first) first = false;
    else std::cout << ", ";
    std::cout << s;
  }
  std::cout << std::endl;
}

int main() {
  std::cout << "Test" << std::endl;
  CCustomOp cop = CustomOp::Register<AddScalarOp>();

  std::cout << "Start..." << std::endl;
  CArgs *args, *tensors;
  CArgs *inshapes, *oshapes;
  CArgs *inames, *onames;

  // Initialize an operator
  CustomOp* op = static_cast<CustomOp*>(cop.init()); 

  // IN/OUT Names
  cop.input_names(op, inames);
  cop.output_names(op, onames);

  // Infer Shape
  cop.infer_shape(op, inshapes, oshapes);

  // Forward and Backward
  cop.forward(op, args, tensors);
  cop.backward(op, args, tensors);

  cop.deleter(op);
  return 0;
}
