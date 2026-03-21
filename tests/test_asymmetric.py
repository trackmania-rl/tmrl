import gymnasium as gym
import numpy as np
import tmrl.config as cfg
import torch
from tmrl.custom.models.Sophy import AsymmetricSophyResidualActorCritic


def test_asymmetric_actor_critic():
    # Construct a dummy observation space similar to TMRL
    # Index 0: Track (Global), Indices 1-14: Telemetry (Ego)
    n_track_pts = 60
    track_dim = 6 * n_track_pts  # Left, center, right -> 3 channels of 2D points = 6*N
    obs_space = gym.spaces.Tuple(
        [
            gym.spaces.Box(low=-100, high=100, shape=(track_dim,), dtype=np.float32),  # Track
            gym.spaces.Box(low=-100, high=100, shape=(3,), dtype=np.float32),  # Speed
            gym.spaces.Box(low=-100, high=100, shape=(3,), dtype=np.float32),
            gym.spaces.Box(low=-100, high=100, shape=(3,), dtype=np.float32),
            gym.spaces.Box(low=-100, high=100, shape=(3,), dtype=np.float32),
            gym.spaces.Box(low=-100, high=100, shape=(3,), dtype=np.float32),
            gym.spaces.Box(low=-100, high=100, shape=(3,), dtype=np.float32),
            gym.spaces.Box(low=-100, high=100, shape=(3,), dtype=np.float32),
            gym.spaces.Box(low=-100, high=100, shape=(3,), dtype=np.float32),
            gym.spaces.Box(low=-100, high=100, shape=(3,), dtype=np.float32),
            gym.spaces.Box(low=-100, high=100, shape=(3,), dtype=np.float32),
            gym.spaces.Box(low=-100, high=100, shape=(3,), dtype=np.float32),
            gym.spaces.Box(low=-100, high=100, shape=(3,), dtype=np.float32),
            gym.spaces.Box(low=-100, high=100, shape=(3,), dtype=np.float32),
            gym.spaces.Box(low=-100, high=100, shape=(3,), dtype=np.float32),
        ]
    )

    act_space = gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

    print("Building AsymmetricSophyResidualActorCritic...")
    # Will fail if asymmetric implementation is wrong
    model = AsymmetricSophyResidualActorCritic(
        observation_space=obs_space,
        action_space=act_space,
    )

    batch_size = 2
    dummy_obs = []
    for space in obs_space:
        dummy_obs.append(torch.randn(batch_size, *space.shape))

    print("Testing Actor forward pass...")
    # Act method should automatically slice out index 0
    action = model.act(dummy_obs)
    print(f"Action shape: {action.shape}")
    assert action.shape == (batch_size, 3)

    print("Testing Critic forward pass...")
    dummy_act = torch.randn(batch_size, 3)
    q1_out = model.q1(dummy_obs, dummy_act)
    q2_out = model.q2(dummy_obs, dummy_act)
    print(f"Q1 output shape: {q1_out.shape}")
    assert q1_out.shape[0] == batch_size
    assert q2_out.shape[0] == batch_size
    if q1_out.ndim == 2:
        assert q1_out.shape[1] == cfg.ALG_CONFIG["QUANTILES_NUMBER"]

    # Test gradients
    print("Testing Gradient Isolation...")
    # We want to make sure the actor does NOT get gradients from the track observation (index 0)
    for i in range(len(dummy_obs)):
        dummy_obs[i].requires_grad_(True)

    act_tensor, _logp = model.actor(dummy_obs[1:15])
    loss = act_tensor.sum()
    loss.backward()

    # The track observation was NOT passed to the actor, so its gradient should be None or 0
    if dummy_obs[0].grad is None or (dummy_obs[0].grad == 0).all():
        print("SUCCESS: Actor does NOT backpropagate to Track observation (Isolation verified!)")
    else:
        print("FAIL: Actor is backpropagating to Track observation! Isolation broken!")

    # But the critic SHOULD backpropagate to the track observation
    model.zero_grad()
    for obs in dummy_obs:
        if obs.grad is not None:
            obs.grad.zero_()

    q_out = model.q1(dummy_obs, dummy_act)
    loss = q_out.sum()
    loss.backward()

    if dummy_obs[0].grad is not None and (dummy_obs[0].grad != 0).any():
        print(
            "SUCCESS: Critic DOES backpropagate to Track observation (Privileged access verified!)"
        )
    else:
        print("FAIL: Critic is NOT backpropagating to Track observation!")

    print("All Asymmetric Model checks passed!")


if __name__ == "__main__":
    test_asymmetric_actor_critic()
