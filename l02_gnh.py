# %%
from typing import Callable
import jax
import jax.numpy as jnp


# %%
weights = jnp.array([[1, 2], [3, 4], [5, 6]], dtype=jnp.float32)
f = lambda x: weights @ x
# x value doesn't matter, as the system is already linear.
x = jnp.array([0, 0], dtype=jnp.float32)
v_in = jnp.array([0, 1], dtype=jnp.float32)
v_out = jnp.array([1, 2, 3], dtype=jnp.float32)

assert jnp.all(jax.jacobian(f)(x) == weights)

y: jnp.ndarray
f_jvp: jnp.ndarray
y, f_jvp = jax.jvp(f, (x,), (v_in,))
assert jnp.all(f(x) == y)
assert jnp.all(f_jvp == weights @ v_in)

y: jnp.ndarray
f_vjp: Callable[[jnp.ndarray], jnp.ndarray]
y, f_vjp = jax.vjp(f, x)
assert jnp.all(f(x) == y)
assert jnp.all(f_vjp(v_out)[0] == v_out @ weights)

# %%
key = jax.random.PRNGKey(0)
w1 = jax.random.normal(key, (5, 2))
w2 = jax.random.normal(key, (3, 5))
nn_fn = lambda x: w2 @ (w1 @ x)  # (x) -> (y)
loss_fn = lambda x: jnp.sum(nn_fn(x))  # (y) -> (1)


def jacobian(
    fn: Callable[[jnp.ndarray], jnp.ndarray],  # (x) -> (1)
    x: jnp.ndarray,  # (x)
) -> jnp.ndarray:  # (x)
    y, fn_vjp = jax.vjp(fn, x)  # (y) -> (x)
    (jacobian,) = fn_vjp(jnp.ones_like(y))
    return jacobian


# def hessian(
#     fn: Callable[[jnp.ndarray], jnp.ndarray],  # (x) -> (1)
#     x: jnp.ndarray,  # (x)
# ) -> jnp.ndarray:  # (x, x)
#     _, fn_vjp = jax.vjp(lambda x: jacobian(fn, x), x)
#     (hessian,) = fn_vjp(jnp.array([1.0, 2.0]))
#     return hessian


def hvp(
    fn: Callable[[jnp.ndarray], jnp.ndarray],  # (x) -> (y)
    x: jnp.ndarray,  # (x)
    y: jnp.ndarray,  # (y)
) -> jnp.ndarray:  # (x)
    _, result = jax.jvp(jax.grad(fn), (x,), (y,))
    return result


def vhp(
    fn: Callable[[jnp.ndarray], jnp.ndarray],  # (x) -> (y)
    x: jnp.ndarray,  # (x)
    y: jnp.ndarray,  # (y)
) -> jnp.ndarray:  # (x)
    _, vjp_fn = jax.vjp(jax.grad(fn), x)
    (result,) = vjp_fn(y)
    return result


print("x_dim", 2)
print("y_dim", 3)
print("jacobian", jacobian(loss_fn, x).shape)
print("hvp", hvp(loss_fn, x, x).shape)
print("vhp", vhp(loss_fn, x, x).shape)
# print("hessian", hessian(loss_fn, x).shape)
# print("jacobian", jax.jacobian(nn_fn)(x).shape)


# def hvp(
#     fn: Callable[[jnp.ndarray], jnp.ndarray],  # (x) -> (y)
#     x: jnp.ndarray,  # (x)
# ) -> Callable[[jnp.ndarray], jnp.ndarray]:
#     def fn_jacobian(x: jnp.ndarray) -> jnp.ndarray:
#         _, jacobian = jax.jvp(fn, (x,), (jnp.eye(x.shape[0]),))
#         return jacobian

#     _, fn_hvp = jax.vjp(fn_jacobian, x)
#     return fn_hvp
#     # return lambda v: jax.jvp(fn_jacobian, (x,), (v,))[1]

#     # _, fn_vjp = jax.vjp(fn, x)  # (y) -> (x)
#     # _, hessian = jax.jvp(fn_vjp, (x,), ([1],))
#     # return hessian


# x = jnp.array([1, 2], dtype=jnp.float32)
# # print(loss_fn(x))
# # v = jnp.array([1, 0], dtype=jnp.float32)
# hvp(loss_fn, x)


# def gnh(
#     nn_fn: Callable[[jnp.ndarray], jnp.ndarray],
#     loss_fn: Callable[[jnp.ndarray], jnp.ndarray],
#     x: jnp.ndarray,
# ) -> jnp.ndarray:
