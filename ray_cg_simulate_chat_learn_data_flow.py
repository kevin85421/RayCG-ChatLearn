import ray
from ray.dag import InputNode
from ray.experimental.channel.torch_tensor_type import TorchTensorType
import torch

@ray.remote(num_gpus=1)
class PolicyInferenceModel:
    def __init__(self):
        pass

    def forward_step(self, data):
        return data

@ray.remote(num_gpus=1)
class ReferenceModel:
    def __init__(self):
        pass

    def forward_step(self, data):
        return data
    
@ray.remote(num_gpus=1)
class RewardModel:
    def __init__(self):
        pass

    def forward_step(self, policy_data, reward_data):
        return [policy_data, reward_data]
    
policy = PolicyInferenceModel.remote()
reference = ReferenceModel.remote()
reward = RewardModel.remote()

with InputNode() as input:
    policy_output = policy.forward_step.bind(input)
    policy_output.with_type_hint(TorchTensorType(transport=TorchTensorType.NCCL))
    reference_output = reference.forward_step.bind(policy_output)
    reference_output.with_type_hint(TorchTensorType(transport=TorchTensorType.NCCL))
    reward_output = reward.forward_step.bind(policy_output, reference_output)
compiled_dag = reward_output.experimental_compile()

def make_experience(query):
    return compiled_dag.execute(query)

def learn(input_data, num_episodes, batch_per_episode):
    for _ in range(num_episodes):
        ref = make_experience(input_data)
        # Simulate the Policy trainer consumes the data
        print(ray.get(ref))

gpu_tensor = torch.tensor([1.0, 2.0, 3.0], device='cuda')
learn(gpu_tensor, 1, 1)