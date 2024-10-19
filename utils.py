import ray
import torch

batch_size = 1024
prompt_seq_len = 4096
num_samples_per_query = 2
bytes_per_int64 = 8
# TODO: not sure which value is appropriate, so use 0 to calculate
# the lower bound of data transfer.
max_new_tokens = 0

# Policy Inference input
policy_inference_input = batch_size * prompt_seq_len * bytes_per_int64

# Policy Inference output (96 MB)
# 1. query + completion
policy_query_completion = (
    batch_size
    * num_samples_per_query
    * (prompt_seq_len + max_new_tokens)
    * bytes_per_int64
)

# 2. logprobs
policy_logprobs = policy_query_completion // 2

# 3. misc
# TODO: need to check the implementation
policy_misc = 0

policy_inference_output = policy_query_completion + policy_logprobs + policy_misc

# Reference model output (64 MB)
reference_output = (
    batch_size
    * num_samples_per_query
    * (prompt_seq_len + max_new_tokens)
    * bytes_per_int64
)

# Reward model output (160 MB)
reward_output = policy_inference_output + reference_output


class Model:
    def __init__(self, output_size_in_bytes):
        bytes_per_int64 = 8
        self.output_data = torch.zeros(
            output_size_in_bytes // bytes_per_int64, dtype=torch.int64, device="cuda"
        )
        # Verify tensor size (bytes)
        assert (
            self.output_data.numel() * self.output_data.element_size()
            == output_size_in_bytes
        )


@ray.remote(num_gpus=1)
class PolicyInferenceModel(Model):
    def forward_step(self, data):
        return self.output_data


@ray.remote(num_gpus=1)
class ReferenceModel(Model):
    def forward_step(self, data):
        return self.output_data


@ray.remote(num_gpus=1)
class RewardModel(Model):
    def forward_step(self, policy_data, reward_data):
        return self.output_data
