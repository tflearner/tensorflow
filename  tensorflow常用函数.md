#  **tensorflow常用函数**

[reference](http://blog.csdn.net/u014595019/article/details/52805444)

## *tf.constant*

```python
tf.constant(value,dtype=None,shape=None,name=’Const’) 
a = tf.constant(2,shape=[2])
b = tf.constant(2,shape=[2,2])
c = tf.constant([1,2,3],shape=[6])
d = tf.constant([1,2,3],shape=[3,2])

sess = tf.InteractiveSession()
print(sess.run(a))
#[2 2]
print(sess.run(b))
#[[2 2]
# [2 2]]
print(sess.run(c))
#[1 2 3 3 3 3]
print(sess.run(d))
#[[1 2]
# [3 3]
# [3 3]]
```

	##  *tf.get_variable*

```python
get_variable(name, shape=None, dtype=dtypes.float32, initializer=None,regularizer=None, trainable=True, collections=None,caching_device=None, partitioner=None, validate_shape=True,custom_getter=None):
```

## *tf.matmul*

```python
matmul(a, b,
           transpose_a=False, transpose_b=False,
           a_is_sparse=False, b_is_sparse=False,
           name=None):
```

​	transpose=TRUE 转置矩阵

​	a_is_sparse 稀疏矩阵

## *tf.reshape*

shape=[-1], 表示要将tensor展开成一个list 
如果 shape=[a,b,c,…] 其中每个a,b,c,..均>0，那么就是常规用法 
如果 shape=[a,-1,c,…] 此时b=-1，a,c,..依然>0。这表示tf会根据tensor的原尺寸，自动计算b的值

```python
# tensor 't' is [1, 2, 3, 4, 5, 6, 7, 8, 9]
# tensor 't' has shape [9]
reshape(t, [3, 3]) ==> [[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]]

# tensor 't' is [[[1, 1], [2, 2]],
#                [[3, 3], [4, 4]]]
# tensor 't' has shape [2, 2, 2]
reshape(t, [2, 4]) ==> [[1, 1, 2, 2],
                        [3, 3, 4, 4]]

# tensor 't' is [[[1, 1, 1],[2, 2, 2]],
#                [[3, 3, 3],[4, 4, 4]],
#                [[5, 5, 5],[6, 6, 6]]]
# tensor 't' has shape [3, 2, 3]
# pass '[-1]' to flatten 't'
reshape(t, [-1]) ==> [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6]

# -1 can also be used to infer the shape
# -1 is inferred to be 9:
reshape(t, [2, -1]) ==> [[1, 1, 1, 2, 2, 2, 3, 3, 3],
                         [4, 4, 4, 5, 5, 5, 6, 6, 6]]

# -1 is inferred to be 2:
reshape(t, [-1, 9]) ==> [[1, 1, 1, 2, 2, 2, 3, 3, 3],
                         [4, 4, 4, 5, 5, 5, 6, 6, 6]]

# -1 is inferred to be 3:
reshape(t, [ 2, -1, 3]) ==> [[[1, 1, 1],
                              [2, 2, 2],
                              [3, 3, 3]],
                             [[4, 4, 4],
                              [5, 5, 5],
                              [6, 6, 6]]]
```

## *tf.gradient*

计算导数

```python
def gradients(ys,
              xs,
              grad_ys=None,
              name="gradients",
              colocate_gradients_with_ops=False,
              gate_gradients=False,
              aggregation_method=None):
```

## *tf.variable_scope*

为变量添加命名域

```python
with tf.variable_scope("foo"):
      with tf.variable_scope("bar"):
          v = tf.get_variable("v", [1])
          assert v.name == "foo/bar/v:0"
```

```python
def variable_scope(name_or_scope, reuse=None, initializer=None,
                   regularizer=None, caching_device=None, partitioner=None,
                   custom_getter=None):
```

name_or_scope: `string` or `VariableScope`: the scope to open. 
reuse: `True` or `None`; if `True`, we [Go](http://lib.csdn.net/base/go) into reuse mode for this scope as well as all sub-scopes; if `None`, we just inherit the parent scope reuse. 如果reuse=True, 那么就是使用之前定义过的name_scope和其中的变量， 
initializer: default initializer for variables within this scope. 
regularizer: default regularizer for variables within this scope. 
caching_device: default caching device for variables within this scope. 
partitioner: default partitioner for variables within this scope. 
custom_getter: default custom getter for variables within this scope.

