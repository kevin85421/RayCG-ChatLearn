import ray
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
    
class Queue:
    """
    This is a simple replacement for `ray.util.queue.Queue` to simulate the
    data flow in ChatLearn.
    """
    def __init__(self):
        self.obj_ref = None

    def put(self, data):
        if isinstance(data, torch.Tensor) and data.is_cuda:
            data = data.cpu()
        elif isinstance(data, ray.ObjectRef):
            # `ray.put` doesn't allow to put ray.ObjectRef directly.
            # This is a bit weird, but ChatLearn actually converts the
            # ObjectRef to a dictionary and then calls `ray.put`. See
            # `generate_step_one_model` for more details.
            data = {"obj_ref": data}
        self.obj_ref = ray.put(data)

    def get(self):
        data = ray.get(self.obj_ref)
        if isinstance(data, dict):
            assert "obj_ref" in data
            data = ray.get(data["obj_ref"])
        if isinstance(data, torch.Tensor) and not data.is_cuda:
            data = data.cuda()
        return data

policy = PolicyInferenceModel.remote()
reference = ReferenceModel.remote()
reward = RewardModel.remote()

input_queue = Queue()
output_queue = Queue()

policy_reference_queue = Queue()
policy_reward_queue = Queue()
reference_reward_queue = Queue()

# Simulate the data flow

def make_experience(query):
    # Add `query` to the queue of the input model (i.e. policy model),
    # and then `get` it. This is confusing, but you can check the
    # implementation in: `Environment.setup_queues` (produce) and
    # `generate_step_one_model_internal` (consume) for more details.
    # Comment out the following two lines to avoid the confusion.

    # input_queue.put(query)
    # query = input_queue.get()

    # Step 1: Policy inference
    ref = policy.forward_step.remote(query)
    for q in [policy_reference_queue, policy_reward_queue]:
        q.put(ref)

    # Step 2: Reference model
    policy_output = policy_reference_queue.get()
    ref = reference.forward_step.remote(policy_output)
    reference_reward_queue.put(ref)

    # Step 3: Reward model
    policy_output = policy_reward_queue.get()
    reference_output = reference_reward_queue.get()
    reward_output = reward.forward_step.remote(policy_output, reference_output)
    output_queue.put(reward_output)

    # Step 4: Consume by the dataloader for training
    return output_queue

def learn(input_data, num_episodes, batch_per_episode):
    for _ in range(num_episodes):
        queue = make_experience(input_data)
        # Simulate the Policy trainer consumes the data
        print(queue.get())

gpu_tensor = torch.tensor([1.0, 2.0, 3.0], device='cuda')

learn(gpu_tensor, 1, 1)
