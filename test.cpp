#include <iostream>
#include "custom_op.h"
using namespace std;

class TArgs {
};

class CustomOp {
public:
  CustomOp() {
    // fill the attributes of CCustomOp
    c_op.forward = nullptr;
    c_op.backward = nullptr;
    c_op.input_names = nullptr;
    c_op.output_names = nullptr;
    c_op.infer_shape = nullptr;
    c_op.init = nullptr;
    c_op.deleter = nullptr;
    c_op.manager_ctx = nullptr;
  }
  virtual void Forward(TArgs* args, TArgs* tensors) = 0;
  virtual void Backward(TArgs* args, TArgs* tensors) = 0;
private:
  CCustomOp c_op;
};

class AddScalarOp : public CustomOp {
public:
  void Forward(TArgs* args, TArgs* tensors) {
  }
  void Backward(TArgs* args, TArgs* tensors) {
  }
private:
};

int main() {
  cout << "Test" << endl;
  return 0;
}
