import ray
from ray.util.queue import Queue

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

input_queue = Queue()
output_queue = Queue()

policy_reference_queue = Queue()
policy_reward_queue = Queue()
reference_reward_queue = Queue()

# Simulate the data flow

def make_experience(query):
    input_queue.put(query)

    # Step 1: Policy inference
    query = input_queue.get()
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

learn("input_data", 1, 1)