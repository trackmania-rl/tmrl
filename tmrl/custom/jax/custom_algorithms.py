import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax import nnx

from tmrl.core.training import TrainingAgent

from tmrl.core.jax.util import get_rngs
import tmrl.custom.jax.custom_models as models


# === Agent ===


class _AlphaModule(nnx.Module):
    """
    Wraps the temperature alpha in an nnx.Module to make it learnable
    """
    def __init__(self, initial_alpha: float):
        self.log_alpha = nnx.Param(jnp.log(jnp.full((1,), initial_alpha, dtype=jnp.float32)))


# Soft Actor-Critic ====================================================================================================


class NNXSACAgent(nnx.Module, TrainingAgent):
    """
    NNX implementation of Soft Actor-Critic (v1 and v2 depending on learn_entropy_coef)
    """
    def __init__(self,
                 observation_space: type,
                 action_space: type,
                 device: str = None,  # device where the model will live (None for auto)
                 model_cls: type = models.NNXREDQMLPActorCritic,
                 gamma: float = 0.99,
                 polyak: float = 0.995,
                 alpha: float = 0.2,  # fixed (v1) or initial (v2) value of the entropy coefficient
                 lr_actor: float = 1e-3,  # learning rate
                 lr_critic: float = 1e-3,  # learning rate
                 lr_entropy: float = 1e-3,  # entropy autotuning (SAC v2)
                 learn_entropy_coef: bool = True,  # if True, SAC v2 is used, else, SAC v1 is used
                 target_entropy: float = None,  # if None, the target entropy for SAC v2 is set automatically
                 optimizer_actor: str = "adam",  # one of ["adam", "adamw", "sgd"]
                 optimizer_critic: str = "adam",  # one of ["adam", "adamw", "sgd"]
                 betas_actor: tuple = None,  # for Adam and AdamW
                 betas_critic: tuple = None,  # for Adam and AdamW
                 l2_actor: float = None,  # weight decay
                 l2_critic: float = None,  # weight decay
                 rngs_seed: int = None,  # used to initialize random generators
                 internal_rngs_seed: int = None  # used to initialize random generators
                 ):
        super().__init__(observation_space, action_space, device)
        self.model_cls = model_cls
        self.gamma = gamma
        self.polyak = polyak
        self.alpha = alpha
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.lr_entropy = lr_entropy
        self.learn_entropy_coef = learn_entropy_coef
        self.target_entropy = target_entropy
        self.optimizer_actor = optimizer_actor
        self.optimizer_critic = optimizer_critic
        self.betas_actor = betas_actor
        self.betas_critic = betas_critic
        self.l2_actor = l2_actor
        self.l2_critic = l2_critic

        # random seeds by default
        self._rngs = get_rngs(internal_rngs_seed, internal_rngs_seed + 1, internal_rngs_seed + 2) if internal_rngs_seed else get_rngs()
        rngs = get_rngs(rngs_seed, rngs_seed + 1, rngs_seed + 2) if rngs_seed else get_rngs()

        # initialize trained model and target model
        model = self.model_cls(observation_space, action_space, rngs=rngs)
        model_target = nnx.clone(model)
        self.actor = model.actor
        self.qs = model.qs
        self.actor_target = model_target.actor
        self.qs_target = model_target.qs

        # optimizers (optax)
        def _make_tx(name: str, lr: float, betas, l2: float):
            name = name.lower()

            optimizer_kwargs = {"learning_rate": lr}
            if name in ["adam", "adamw"] and betas is not None:
                optimizer_kwargs["b1"], optimizer_kwargs["b2"] = tuple(betas)

            if name == "adamw":
                optimizer_kwargs["weight_decay"] = l2 if l2 is not None else 0.0
                optimizer = optax.adamw(**optimizer_kwargs)
            else:
                if name == "adam":
                    optimizer_cls = optax.adam
                else:
                    optimizer_cls = optax.sgd
                optimizer = optimizer_cls(**optimizer_kwargs)
                if l2 is not None:
                    optimizer = optax.chain(optimizer, optax.add_decayed_weights(l2))
            
            return optimizer
        
        tx_actor = _make_tx(name=optimizer_actor, lr=lr_actor, betas=betas_actor, l2=l2_actor)
        tx_critic = _make_tx(name=optimizer_critic, lr=lr_critic, betas=betas_critic, l2=l2_critic)

        self.pi_optimizer = nnx.Optimizer(self.actor, tx_actor, wrt=nnx.Param)
        self.q_optimizer = nnx.Optimizer(self.qs, tx_critic, wrt=nnx.Param)

        if self.target_entropy is None:
            self.target_entropy = -np.prod(action_space.shape)
        else:
            self.target_entropy = float(self.target_entropy)
        
        self.alpha_module = _AlphaModule(self.alpha)
        tx_alpha = optax.adam(self.lr_entropy) if self.learn_entropy_coef else optax.set_to_zero()
        self.alpha_optimizer = nnx.Optimizer(self.alpha_module, tx_alpha, wrt=nnx.Param)

    def get_actor(self):
        return self.actor

    @nnx.jit
    def train(self, batch):

        o, a, r, o2, d, _ = batch

        # Dynamic from self:
        actor = self.actor
        qs = self.qs
        actor_target = self.actor_target
        qs_target = self.qs_target
        pi_optimizer = self.pi_optimizer
        q_optimizer = self.q_optimizer
        alpha_module = self.alpha_module
        alpha_optimizer = self.alpha_optimizer
        rngs = self._rngs

        # Static from self:
        gamma = self.gamma
        polyak = self.polyak
        target_entropy = self.target_entropy
        learn_entropy_coef = self.learn_entropy_coef

        alpha_t = jnp.exp(alpha_module.log_alpha[0])

        # Critic update
        def loss_fn_q(qs: nnx.List, rngs: nnx.Rngs):
            a2, logp_a2 = actor(o2, rngs=rngs)
            q1_pi_targ = qs_target[0](o2, a2)
            q2_pi_targ = qs_target[1](o2, a2)
            q_pi_targ = jnp.minimum(q1_pi_targ, q2_pi_targ)
            backup = jax.lax.stop_gradient(r + gamma * (1 - d) * (q_pi_targ - alpha_t * logp_a2))  # stop_gradient is mostly here for readibility
            q1 = qs[0](o, a)
            q2 = qs[1](o, a)
            loss_q1 = jnp.mean((q1 - backup) ** 2)  # MSE
            loss_q2 = jnp.mean((q2 - backup) ** 2)  # MSE
            return (loss_q1 + loss_q2) / 2.0
        loss_q, q_grads = nnx.value_and_grad(loss_fn_q)(qs, rngs)
        q_optimizer.update(qs, q_grads)

        # Actor update
        def loss_fn_pi(actor: nnx.Module, rngs: nnx.Rngs):
            pi, logp_pi = actor(o, rngs=rngs)
            q1_pi = qs[0](o, pi)
            q2_pi = qs[1](o, pi)
            q_pi = jnp.minimum(q1_pi, q2_pi)
            loss_pi = jnp.mean(alpha_t * logp_pi - q_pi)
            return loss_pi, logp_pi
        (loss_pi, logp_pi), pi_grads = nnx.value_and_grad(loss_fn_pi, has_aux=True)(actor, rngs)
        pi_optimizer.update(actor, pi_grads)

        # Entropy coeffient update
        loss_alpha = jnp.zeros(())  # zero scalar (for logging when learn_entropy_coef is False)
        if learn_entropy_coef:
            def loss_fn_alpha(alpha_module: _AlphaModule):
                return -jnp.mean(alpha_module.log_alpha[0] * jax.lax.stop_gradient(logp_pi + target_entropy))
            loss_alpha, alpha_grads = nnx.value_and_grad(loss_fn_alpha)(alpha_module)
            alpha_optimizer.update(alpha_module, alpha_grads)
            alpha_t = jnp.exp(alpha_module.log_alpha[0])
        
        # Polyak slow target update
        for original, target in ((actor, actor_target), (qs, qs_target)):
            original_params = nnx.state(original, nnx.Param)
            target_params = nnx.state(target, nnx.Param)
            new_target_params = jax.tree.map(
                lambda pt, p: polyak * pt + (1.0 - polyak) * p,  # f
                target_params,  # tree
                original_params,  # rest
            )
            nnx.update(target, new_target_params)
        
        ret_dict = dict(
            loss_actor=loss_pi,
            loss_critic=loss_q,
        )
        if learn_entropy_coef:
            ret_dict["loss_entropy_coef"] = loss_alpha
            ret_dict["entropy_coef"] = alpha_t

        return ret_dict


if __name__ == "__main__":
    import gymnasium as gym
    env = gym.make("Pendulum-v1")
    agent = NNXSACAgent(observation_space=env.observation_space, action_space=env.action_space)

    o, _ = env.reset()
    for _ in range(3):
        a = env.action_space.sample()
        o2, r, terminated, truncated, info = env.step(a)

        batch_size = 256

        agent.train(
            (
                jnp.array([o] * batch_size),
                jnp.array([a] * batch_size),
                jnp.array([r] * batch_size),
                jnp.array([o2] * batch_size),
                jnp.array([terminated] * batch_size),
                jnp.array([truncated] * batch_size)
            )
        )
        o = o2
        print(r)
        if terminated or truncated:
            break
