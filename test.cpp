#include "custom_op.hpp"

class AddScalarOp : public CustomOp {
public:
  AddScalarOp() {
    std::cout << "Create AddScalarOp" << std::endl;
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
  CustomOp* op = static_cast<CustomOp*>(cop.creator()); 

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
