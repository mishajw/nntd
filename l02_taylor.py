# %%
from tqdm import tqdm
from itertools import product
import pandas as pd
import plotly.express as px
import jax
from jaxtyping import Float, jaxtyped
import jax.numpy as jnp
from typeguard import typechecked
import plotly.figure_factory as ff


# %%
"""
Let's build a simple linear regression model.
"""


@jaxtyped(typechecker=typechecked)
def lr(
    inputs: Float[jax.Array, "data feature"],
    outputs: Float[jax.Array, "data"],
    weights: Float[jax.Array, "feature"],
) -> Float[jax.Array, ""]:
    outputs_pred = jnp.dot(inputs, weights)
    return jnp.mean((outputs - outputs_pred) ** 2)


key = jax.random.key(42)
inputs_key, key = jax.random.split(key)
inputs = jax.random.normal(inputs_key, (16, 2))
noise_key, key = jax.random.split(key)
noise = jax.random.uniform(noise_key, (16,)) * 0.1
outputs = jnp.dot(inputs, jnp.array([1, 2])) + noise
weights_key, key = jax.random.split(key)
weights = jax.random.normal(weights_key, (2,))


# %%
"""
The vector-Jacobian product (VJP) computes the product of a vector & the Jacobian of a
function.

The `vjp` function returns a function that takes the *vector* and computes the VJP.

This can be used to linearize the function around a point, by using Taylor
approximations.

N.B.: The VJP computes $vJ$, where $J$ has the shape (num_outputs, num_inputs). Thus,
$v$ must have the shape (num_outputs,). That's how we know that the input to the VJP is
the outputs.
"""
ans, lr_vjp = jax.vjp(lr, inputs, outputs, weights)
assert (ans == lr(inputs, outputs, weights)).all()

# %%
"""
We can use the VJP to compute the gradient of the function with respect to the inputs,
by simply passing a vector of ones.
"""


@jaxtyped(typechecker=typechecked)
def lr_grad(
    inputs: Float[jax.Array, "data feature"],
    outputs: Float[jax.Array, "data"],
    weights: Float[jax.Array, "feature"],
) -> Float[jax.Array, "feature"]:
    _, lr_vjp = jax.vjp(lr, inputs, outputs, weights)
    _, _, weights_grad = lr_vjp(1.0)
    return weights_grad


assert (
    lr_grad(inputs, outputs, weights)
    == jax.grad(lr, argnums=[2])(inputs, outputs, weights)[0]
).all()

# %%
"""
We can also use the VJP to compute directional gradients, but first we need the *JVP*.
This is simple to get by invoking the VJP twice.
"""


@jaxtyped(typechecker=typechecked)
def lr_jvp(
    inputs: Float[jax.Array, "data feature"],
    outputs: Float[jax.Array, "data"],
    weights: Float[jax.Array, "feature"],
):
    ans, lr_vjp = jax.vjp(lr, inputs, outputs, weights)
    # Zeros, as we want the vjp around the zero-point.
    _, f_vjp_vjp = jax.vjp(lr_vjp, jnp.zeros_like(ans))

    @jaxtyped(typechecker=typechecked)
    def jvp_fn(
        inputs: Float[jax.Array, "data feature"],
        outputs: Float[jax.Array, "data"],
        weights: Float[jax.Array, "feature"],
    ) -> Float[jax.Array, ""]:
        return f_vjp_vjp((inputs, outputs, weights))[0]

    return jvp_fn


assert (
    lr_jvp(inputs, outputs, weights)(inputs, outputs, weights)
    == jax.jvp(lr, (inputs, outputs, weights), (inputs, outputs, weights))[1]
)

# %%
learning_rate = 0.1
df = []
weights = jax.random.normal(weights_key, (2,))
for i in tqdm(range(100)):
    weights_grad = lr_grad(inputs, outputs, weights)
    weights = weights - learning_rate * weights_grad
    loss = lr(inputs, outputs, weights)
    log_loss = jnp.log(loss)
    directional_grad = lr_jvp(inputs, outputs, weights)(inputs, outputs, weights_grad)

    w1, w2 = weights
    w1_grad, w2_grad = weights_grad
    df.append(
        dict(
            i=i,
            w1=w1,
            w2=w2,
            w1_grad=w1_grad,
            w2_grad=w2_grad,
            loss=loss.item(),
            log_loss=log_loss.item(),
            directional_grad=directional_grad.item(),
            src="train",
        )
    )

for w1, w2 in tqdm(list(product(jnp.linspace(0, 2, 20), jnp.linspace(1, 3, 20)))):
    weights = jnp.array([w1, w2], dtype=jnp.float32)
    loss = lr(inputs, outputs, weights)
    log_loss = jnp.log(loss)
    w1_grad, w2_grad = lr_grad(inputs, outputs, weights)
    directional_grad = lr_jvp(inputs, outputs, weights)(
        inputs, outputs, jnp.array([1, 0], dtype=jnp.float32)
    )
    df.append(
        dict(
            w1=w1,
            w2=w2,
            w1_grad=w1_grad,
            w2_grad=w2_grad,
            loss=loss.item(),
            log_loss=log_loss.item(),
            directional_grad=directional_grad.item(),
            src="grid",
        )
    )


df = pd.DataFrame(df)

# %%
px.scatter(df, x="w1", y="w2", color="log_loss")

# %%
px.scatter(df, x="loss", y="directional_grad")

# %%
df2 = df  # .query("i % 5 == 0")
ff.create_quiver(
    x=df2["w1"],
    y=df2["w2"],
    u=-df2["w1_grad"],
    v=-df2["w2_grad"],
    scale=learning_rate,
)
