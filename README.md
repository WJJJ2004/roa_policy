roa_policy_driver

roa_policy_driver는 ROA 보행 정책(ONNX)을 C++에서 실행하기 위한 순수 라이브러리이다.
ROS2 의존성이 없으며, ros2_control/controller/driver에서 안전하게 호출할 수 있도록 설계되었다.

ONNX Runtime 기반 CPU inference

모델 입출력 차원 검증

실시간 루프에서 호출 가능한 run(obs -> action) API 제공

golden test(시뮬레이션에서 덤프한 obs/action)로 결과 일치성 검증

Requirements

Ubuntu 22.04

C++17

ONNX Runtime 1.23.2 (prebuilt tgz, CPU-only)

설치 경로: /opt/onnxruntime/current

헤더: /opt/onnxruntime/current/include/onnxruntime_cxx_api.h

라이브러리: /opt/onnxruntime/current/lib/libonnxruntime.so

Repository Layout

ang_vel_observation/model_74948/policy.onnx : 기본 테스트 대상 모델

include/roa_policy_driver/policy_driver.hpp : public API

src/policy_driver.cpp : 구현

tests/ : dimension/golden/stress 테스트

Build
cd roa_policy_driver
rm -rf build
mkdir build && cd build
cmake .. -DROA_BUILD_TESTS=ON
cmake --build . -j

Run Tests
cd build
ctest --output-on-failure


테스트 구성:

test_dimensions: 모델 로드 및 input/output dim 검증

test_golden_io: golden_obs.bin을 입력했을 때 golden_action.bin과 동일 출력인지 검증

test_stress: 반복 실행 안정성(메모리/세션) 검증

Install
cd build
sudo cmake --install .
sudo ldconfig


설치 후 외부 CMake 프로젝트에서 find_package(roa_policy_driver)로 사용할 수 있다.

CMake Integration (Consumer)
find_package(roa_policy_driver REQUIRED)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE roa::roa_policy_driver)

API Usage

헤더: #include <roa_policy_driver/policy_driver.hpp>

기본 사용 흐름:

PolicyDriver 생성

load(model_path, options)로 ONNX 로드

실시간 루프에서 run(obs, action) 호출

Example
#include <array>
#include <iostream>
#include <roa_policy_driver/policy_driver.hpp>

int main() {
  roa::policy::PolicyDriver driver;
  roa::policy::Options opt;  // 기본 옵션

  const std::string model_path =
      "ang_vel_observation/model_74948/policy.onnx";

  if (!driver.load(model_path, opt)) {
    std::cerr << "Failed to load model: " << model_path << "\n";
    return 1;
  }

  const int in_dim = driver.input_dim();
  const int out_dim = driver.output_dim();

  std::vector<float> obs(in_dim, 0.0f);
  std::vector<float> action(out_dim, 0.0f);

  // obs 채우기 (사용자 로직)
  // obs = [cmd(3), q_rel(13), qd_rel(13), imu_omega(3), last_action(13)] (예시)

  if (!driver.run(obs.data(), in_dim, action.data(), out_dim)) {
    std::cerr << "Policy inference failed\n";
    return 2;
  }

  std::cout << "action[0]=" << action[0] << "\n";
  return 0;
}

Realtime Usage Notes

load()는 초기화 단계에서 1회만 호출한다 (configure 단계)

제어 루프(update)에서는 run()만 호출한다

성공 경로에서 동적 할당이 없도록(obs/action 버퍼는 외부에서 고정 크기로 유지 권장)

cmd/last_action 등의 입력은 controller 측에서 realtime-safe하게 관리한다

Updating the Policy Model

새 모델로 교체할 때는 다음을 함께 갱신해야 한다.

모델 파일 경로

golden 테스트 데이터:

tests/data/golden_obs.bin

tests/data/golden_action.bin

golden 데이터는 “해당 모델 + 해당 obs 정의” 조합에서 생성된 값이어야 한다.