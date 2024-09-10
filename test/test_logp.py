import unittest
import numpy as np

import jax
from jax.scipy.stats import norm
import jax.numpy as jnp

# 定义函数
def logp_fun_0(x, n, dim):
    return norm.logpdf(x).sum()

def logp_fun_1(x, n, dim):
    i, j = jnp.triu_indices(n, k=1)
    r_ee = jnp.linalg.norm((jnp.reshape(x, (n, 1, dim)) - jnp.reshape(x, (1, n, dim)))[i,j], axis=-1)
    v_ee = jnp.sum(1/r_ee)
    return jnp.sum(x**2) + v_ee

# 测试类
class TestFunctions(unittest.TestCase):
    def setUp(self):
        # 初始化测试数据
        self.n = 3
        self.dim = 2
        self.x = jnp.array(np.random.randn(100, self.n * self.dim))

    def test_logp_fun_0_output(self):
        # 测试 logp_fun_0 的输出维度
        result = jax.vmap(logp_fun_0, (0, None, None))(self.x.reshape(100, self.n, self.dim), self.n, self.dim)
        print(result.shape)

    def test_logp_fun_1_output(self):
        # 测试 logp_fun_1 的输出维度
        result = jax.vmap(logp_fun_1, (0, None, None))(self.x.reshape(100, self.n, self.dim), self.n, self.dim)
        print(result.shape)

# 运行测试
if __name__ == '__main__':
    unittest.main()
