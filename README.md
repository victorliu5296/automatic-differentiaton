I will explain automatic differentiation in my own words because many explanations appear to be confusing, with a lot of people spreading information that isn't accurate. Nonetheless, there are some good resources about this topic:

- A paper explaining how symbolic differentiation and automatic differentiation are the same thing, just with a different data structure representation: https://arxiv.org/pdf/1904.02990

- Wikipedia's entry on automatic differentiation: https://en.wikipedia.org/wiki/Automatic_differentiation

Now, let's delve into my interpretation:

In mathematical terms, we break down the function to be differentiated into a composition of basic functions, such as $\sin(x)$, $x+1$, $2x$.
Then, we leverage the chain rule to obtain a multiplication of these basic functions rather than having to deal with the mess that symbolic differentiation would bring.

### Example

Consider the function:

$$f(x, y):=e^x \cos(y) + x^y$$

To differentiate this function, we use the chain rule:

$$\frac{\partial f(x, y)}{\partial x}=\frac{\partial f(x, y)}{\partial u_n}\frac{\partial u_n}{\partial u_{n-1}}...\frac{\partial u_1}{\partial x}$$

$$\frac{\partial f(x, y)}{\partial y}=\frac{\partial f(x, y)}{\partial u_n}\frac{\partial u_n}{\partial u_{n-1}}...\frac{\partial u_1}{\partial y}$$

Here, the functions $u_1$, $u_2$, ..., $u_n$ are just placeholders for the composed basic functions. We can determine the order of these basic functions by applying a substitution trick using the standard order of operations, similar to the "u-sub" technique in integration.

### Forward and Reverse Mode

Automatic differentiation has two "flavors": forward mode and reverse mode, indicating the direction of traversal in the chain rule/computational graph.

- **Forward mode** calculates in the right-to-left order: $$\frac{\partial u_1}{\partial x}, \frac{\partial u_2}{\partial u_1}, \ldots, \frac{\partial u_n}{\partial u_{n-1}}$$ using the recursive relation:
  $$\frac{\partial u_i}{\partial x} = \frac{\partial u_i}{\partial u_{i-1}} \frac{\partial u_{i-1}}{\partial x}$$
  with $u_n = f(x, y)$.

- **Reverse mode** calculates in the left-to-right order: $$\frac{\partial f(x, y)}{\partial u_n}, \frac{\partial u_n}{\partial u_{n-1}}, \ldots, \frac{\partial u_1}{\partial x}$$ using the recursive relation:
  $$\frac{\partial f(x, y)}{\partial u_i} = \frac{\partial f(x, y)}{\partial u_{i+1}} \frac{\partial u_{i+1}}{\partial u_i}$$
  with $u_0$ being the independent variable used as the "seed".

Forward mode is faster when there are more outputs than inputs, while reverse mode is faster when there are more inputs than outputs, which is often the case in machine learning.


Let's break down the function layer by layer:
$f(x, y)=u_1 \cos(y) + x^y$
where $u_1:=e^x, \frac{\partial u_1}{\partial x}$
$f(x, y)=u_1 u_2 + x^y$
where $u_2:=\cos(y)$
$f(x, y)=u_1 u_2 + u_3$
where $u_3:=x^y$

Now, we are ready to explore the forward and reverse mode automatic differentiation processes.

### Forward Mode

In forward mode, we propagate derivatives from the inputs to the output. We compute the derivatives of these intermediate variables with respect to $x$ and $y$:

#### Derivatives with respect to $x$:

$$\frac{\partial u_1}{\partial x} = e^x$$
$$\frac{\partial u_2}{\partial x} = 0$$ (since $u_2$ depends only on $y$)
$$\frac{\partial u_3}{\partial x} = y x^{y-1}$$

Using the chain rule, we get:

$$\frac{\partial f}{\partial x} = \frac{\partial f}{\partial u_1} \frac{\partial u_1}{\partial x} + \frac{\partial f}{\partial u_3} \frac{\partial u_3}{\partial x}$$

Since $f = u_1 u_2 + u_3$:

$$\frac{\partial f}{\partial u_1} = u_2$$
$$\frac{\partial f}{\partial u_3} = 1$$

Thus:

$$\frac{\partial f}{\partial x} = u_2 \cdot e^x + 1 \cdot y x^{y-1}$$
$$\frac{\partial f}{\partial x} = \cos(y) e^x + y x^{y-1}$$

#### Derivatives with respect to $y$:

$$\frac{\partial u_1}{\partial y} = 0$$ (since $u_1$ depends only on $x$)
$$\frac{\partial u_2}{\partial y} = -\sin(y)$$
$$\frac{\partial u_3}{\partial y} = x^y \ln(x)$$

Using the chain rule, we get:

$$\frac{\partial f}{\partial y} = \frac{\partial f}{\partial u_2} \frac{\partial u_2}{\partial y} + \frac{\partial f}{\partial u_3} \frac{\partial u_3}{\partial y}$$

Since $f = u_1 u_2 + u_3$:

$$\frac{\partial f}{\partial u_2} = u_1$$
$$\frac{\partial f}{\partial u_3} = 1$$

Thus:

$$\frac{\partial f}{\partial y} = u_1 \cdot (-\sin(y)) + 1 \cdot x^y \ln(x)$$
$$\frac{\partial f}{\partial y} = e^x (-\sin(y)) + x^y \ln(x)$$
$$\frac{\partial f}{\partial y} = -e^x \sin(y) + x^y \ln(x)$$

### Reverse Mode

In reverse mode, we propagate derivatives from the output to the inputs. We start by computing the derivatives of the output with respect to the intermediate variables and then propagate these derivatives backward.

1. $$u_1 = e^x$$
2. $$u_2 = \cos(y)$$
3. $$u_3 = x^y$$
4. $$f(x, y) = u_1 u_2 + u_3$$

#### Derivatives of $f$ with respect to intermediate variables:

$$\frac{\partial f}{\partial u_1} = u_2$$
$$\frac{\partial f}{\partial u_2} = u_1$$
$$\frac{\partial f}{\partial u_3} = 1$$

#### Propagate derivatives backward:

$$\frac{\partial u_1}{\partial x} = e^x$$
$$\frac{\partial u_2}{\partial y} = -\sin(y)$$
$$\frac{\partial u_3}{\partial x} = y x^{y-1}$$
$$\frac{\partial u_3}{\partial y} = x^y \ln(x)$$

Using the chain rule, we get:

$$\frac{\partial f}{\partial x} = \frac{\partial f}{\partial u_1} \frac{\partial u_1}{\partial x} + \frac{\partial f}{\partial u_3} \frac{\partial u_3}{\partial x}$$
$$\frac{\partial f}{\partial x} = u_2 \cdot e^x + 1 \cdot y x^{y-1}$$
$$\frac{\partial f}{\partial x} = \cos(y) e^x + y x^{y-1}$$

$$\frac{\partial f}{\partial y} = \frac{\partial f}{\partial u_2} \frac{\partial u_2}{\partial y} + \frac{\partial f}{\partial u_3} \frac{\partial u_3}{\partial y}$$
$$\frac{\partial f}{\partial y} = u_1 \cdot (-\sin(y)) + 1 \cdot x^y \ln(x)$$
$$\frac{\partial f}{\partial y} = e^x (-\sin(y)) + x^y \ln(x)$$
$$\frac{\partial f}{\partial y} = -e^x \sin(y) + x^y \ln(x)$$

Both forward and backward modes yield the same results for the partial derivatives:

$$\frac{\partial f}{\partial x} = \cos(y) e^x + y x^{y-1}$$
$$\frac{\partial f}{\partial y} = -e^x \sin(y) + x^y \ln(x)$$