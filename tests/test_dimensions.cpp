#include "roa_policy_driver/policy_driver.hpp"
#include <cstdlib>
#include <iostream>
#include <string>

static std::string default_model_path() {
  // 현재 repo 트리 기준: ./ang_vel_observation/model_74948/policy.onnx
  return std::string("ang_vel_observation/model_74948/policy.onnx");
}

int main(int argc, char** argv) {
  std::string model = default_model_path();
  if (argc >= 2) model = argv[1];

  roa::policy::PolicyDriver drv;
  if (!drv.load(model)) {
    std::cerr << "FAILED: load model: " << model << "\n";
    return 1;
  }

  std::cout << "Loaded model: " << model << "\n";
  std::cout << "Input name: " << drv.input_name() << " dim=" << drv.input_dim() << "\n";
  std::cout << "Output name: " << drv.output_name() << " dim=" << drv.output_dim() << "\n";

  // sim2sim contract (현재 네 레퍼런스)
  if (drv.input_dim() != 45) {
    std::cerr << "FAILED: expected input_dim=45 but got " << drv.input_dim() << "\n";
    return 2;
  }
  if (drv.output_dim() != 13) {
    std::cerr << "FAILED: expected output_dim=13 but got " << drv.output_dim() << "\n";
    return 3;
  }

  std::cout << "PASS\n";
  return 0;
}

