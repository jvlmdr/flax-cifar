from typing import Any, Iterator, Mapping, Tuple

from jax import numpy as jnp
from jax import tree_util

Tree = Any


def not_mapping(x: Any) -> bool:
    return not isinstance(x, Mapping)


def tree_shape(tree: Tree) -> Tree:
    return tree_util.tree_map(jnp.shape, tree)


def tree_dot(a: Tree, b: Tree) -> Tree:
    return tree_util.tree_reduce(
        jnp.add,
        tree_util.tree_map(lambda x, y: jnp.sum(jnp.multiply(x, y)), a, b))


def tree_l2_norm(tree: Tree, squared: bool = False) -> jnp.ndarray:
    norm2 = tree_util.tree_reduce(jnp.add, tree_util.tree_map(lambda x: jnp.sum(jnp.square(x)), tree))
    if squared:
        return norm2
    else:
        return jnp.sqrt(norm2)


def dict_tree_format(tree: Tree):
    for context, value in dict_tree_items(tree):
        yield '{:s}: {}'.format('.'.join(context), value)


def dict_tree_items(tree: Tree, context: Tuple = ()) -> Iterator[Tuple[Tuple, Any]]:
    if isinstance(tree, Mapping):
        for k, v in tree.items():
            yield from dict_tree_items(v, context=(*context, k))
    else:
        yield context, tree
