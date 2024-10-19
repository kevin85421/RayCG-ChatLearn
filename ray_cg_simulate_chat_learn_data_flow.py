import ray
from ray.dag import InputNode
from ray.experimental.channel.torch_tensor_type import TorchTensorType
import torch
from utils import (
    PolicyInferenceModel,
    ReferenceModel,
    RewardModel,
    policy_inference_input,
    bytes_per_int64,
    policy_inference_output,
    reference_output,
    reward_output,
)
import time


if __name__ == "__main__":
    ray.init()

    policy = PolicyInferenceModel.remote(policy_inference_output)
    reference = ReferenceModel.remote(reference_output)
    reward = RewardModel.remote(reward_output)

    with InputNode() as input:
        policy_output = policy.forward_step.bind(input)
        policy_output.with_type_hint(TorchTensorType(transport=TorchTensorType.NCCL))
        reference_output = reference.forward_step.bind(policy_output)
        reference_output.with_type_hint(TorchTensorType(transport=TorchTensorType.NCCL))
        reward_output = reward.forward_step.bind(policy_output, reference_output)
    compiled_dag = reward_output.experimental_compile()

    def make_experience(query):
        return compiled_dag.execute(query)

    def learn(input_data, num_episodes):
        for _ in range(num_episodes):
            ref = make_experience(input_data)
            # Simulate the Policy trainer consumes the data
            # TODO (kevin85421): https://github.com/ray-project/ray/issues/46440#issuecomment-2362415025
            reward_output = ray.get(ref)
            assert (
                reward_output.numel() * reward_output.element_size()
                == 160 * 1024 * 1024
            )  # 160 MB

    gpu_tensor = torch.zeros(
        policy_inference_input // bytes_per_int64, dtype=torch.int64, device="cuda"
    )
    assert gpu_tensor.numel() * gpu_tensor.element_size() == 32 * 1024 * 1024  # 32 MB

    start_time = time.perf_counter()
    learn(gpu_tensor, 100)
    end_time = time.perf_counter()
    print(f"Execution time: {end_time - start_time} seconds")
