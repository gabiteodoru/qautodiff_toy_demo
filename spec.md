# Homework: Implement Automatic Differentiation in q/kdb+

## Objective

Build a system that automatically computes derivatives of q functions using reverse-mode automatic differentiation (the same algorithm as backpropagation in neural networks).

## Background

Automatic differentiation (AD) computes exact derivatives by applying the chain rule to elementary operations. It's more accurate than numerical differentiation and more efficient than symbolic differentiation for complex functions.

### Why Reverse-Mode?

For a function with many inputs and one output (like neural network loss functions), reverse-mode AD computes all gradients in just **two passes**:
1. **Forward pass**: Compute and store intermediate values
2. **Backward pass**: Traverse computation backward, accumulating gradients

This is O(n) where n is the number of operations, compared to O(n²) for naive symbolic differentiation.

## Requirements

### Core Functionality

Implement a function `AD` with the signature:
```q
AD:{[f;x] ... }
```

Where:
- `f` is a unary q function (takes one parameter `x`)
- `x` is a numeric value (the point at which to evaluate the derivative)
- Returns the derivative `f'(x)` as a float

### Required Operations Support

Your implementation must support these operations:

**Binary operations:**
- Addition: `+`
- Subtraction: `-`
- Multiplication: `*`
- Division: `%`
- Power: `xexp[base;exponent]`

**Unary operations:**
- Exponential: `exp`
- Natural logarithm: `log`
- Negation: `neg`

**Constants:**
- Numeric literals should be treated as constants with zero gradient

### Test Cases

Your implementation must pass these tests:

```q
/ Basic operations
AD[{x*x}; 3] ~ 6f                    / x² → 2x, at x=3: 2*3=6
AD[{x+x}; 5] ~ 2f                    / 2x → 2
AD[{x-x}; 3] ~ 0f                    / x-x = 0 → 0
AD[{x%x}; 3] ~ 0f                    / x/x = 1 → 0

/ Composed operations (remember q is right-to-left!)
AD[{(x*x)-x}; 3] ~ 5f                / x²-x → 2x-1, at x=3: 5
AD[{(x+x)*(x+x)}; 3] ~ 24f           / 4x² → 8x, at x=3: 24
AD[{x%(x+x)}; 3] ~ 0f                / x/(2x) = 1/2 → 0

/ Power functions
AD[{xexp[x;2]}; 3] ~ 6f              / x² → 2x
AD[{xexp[x;3]}; 2] ~ 12f             / x³ → 3x², at x=2: 12
AD[{xexp[2;x]}; 3] ~ 5.545177        / 2^x → 2^x·ln(2)

/ Unary functions
AD[{exp x}; 0] ~ 1f                  / e^x → e^x, at x=0: 1
AD[{log x}; 2] ~ 0.5                 / ln(x) → 1/x, at x=2: 0.5
AD[{neg x}; 3] ~ -1f                 / -x → -1

/ Complex compositions
AD[{exp x*x}; 1] ~ 5.436564          / e^(x²) → 2x·e^(x²)
AD[{log xexp[x;2]}; 2] ~ 1f          / ln(x²) → 2/x, at x=2: 1
AD[{(x*x)%x}; 3] ~ 1f                / x²/x = x → 1
```

### Partial Evaluation

The function should support q's natural projection syntax:

```q
grad_f: AD[{x*x}]       / Create gradient function
grad_f[3]               / Evaluate at x=3, returns 6f
grad_f each 1 2 3 4     / Evaluate at multiple points
```

## Implementation Approach

### Recommended Strategy

1. **Parse the function**: Extract the AST (abstract syntax tree)
   - Hint: q functions can be introspected using `value`
   - Hint: Use `parse` to convert q code strings into AST

2. **Build a computation graph**:
   - Create nodes representing values (variables, constants, intermediate results)
   - Each node should track:
     - Its computed value
     - The operation that created it
     - Its parent nodes (dependencies)
     - Values needed for backward pass

3. **Forward pass**: Execute the function while building the graph
   - Start with input variable node
   - For each operation, create a new node
   - Store intermediate values

4. **Backward pass**: Traverse graph in reverse to compute gradients
   - Start with gradient of output = 1
   - For each node, compute gradients of its parents using chain rule
   - Accumulate gradients (important: same node may be used multiple times!)

5. **Return the input gradient**: Extract gradient of the input variable

### Differentiation Rules Reference

You'll need to implement these rules (chain rule for each operation):

**Binary operations** - compute gradients for both inputs:
- Addition: `d/dx(a+b) = 1, d/dy(a+b) = 1`
- Subtraction: `d/dx(a-b) = 1, d/dy(a-b) = -1`
- Multiplication: `d/dx(a*b) = b, d/dy(a*b) = a` (product rule)
- Division: `d/dx(a/b) = 1/b, d/dy(a/b) = -a/b²` (quotient rule)
- Power: `d/dx(a^b) = b·a^(b-1), d/dy(a^b) = a^b·ln(a)`

**Unary operations** - compute gradient for single input:
- Exponential: `d/dx(e^x) = e^x`
- Logarithm: `d/dx(ln(x)) = 1/x`
- Negation: `d/dx(-x) = -1`

**Chain rule**: If `z = f(y)` and `y = g(x)`, then `dz/dx = (dz/dy) · (dy/dx)`

### Hints

1. **AST structure**:
   - Variables are symbols (type -11h)
   - Constants are numeric atoms (types -9h to -5h)
   - Unary operations: 2-element lists `(operator; operand)`
   - Binary operations: 3-element lists `(operator; left; right)`
   - Operators are functions (type 102h)

2. **Operator types - IMPORTANT**:
   - When you `parse"x*x"`, the `*` in the AST is the **actual function** (type 102h), not a symbol!
   - Check: `type first parse"2*3"` returns `102h` (function)
   - Compare: `type ``*` returns `-11h` (symbol), `type"*"` returns `-10h` (char)
   - This means you can call it directly: `op[val1;val2]` where `op` is from the AST
   - For your ops dictionary, use the function itself as the key: `ops[*]:{...}` not `ops[``*]`

3. **Functional approach**:
   - Use q's `over` (`/`) to fold gradients through the graph
   - Avoid global state - pass data through function parameters
   - Build up the nodes list incrementally

4. **Building the nodes list - WATCH OUT**:
   - When you append dicts with same keys, q converts to a table automatically!
   - This is fine! Tables are lists of dicts in q (they're equivalent)
   - BUT: using `nodes,:enlist dict` can cause type errors when columns have incompatible types
   - **Solution**: Use `nodes: nodes,enlist dict` instead - creates new list, avoids type issues
   - The `,` operator is more forgiving than `,:` for heterogeneous data

5. **Accumulation**:
   - When a variable appears multiple times (like `x*x`), gradients must be **summed**
   - Use `+:` to accumulate gradients for each parent

6. **Operator dictionary**:
   - Store differentiation rules in a dictionary keyed by operator
   - Makes it easy to look up the rule during backward pass
   - Remember: keys are the actual functions, not symbols!

## Evaluation Criteria

Your solution will be evaluated on:

1. **Correctness**: Passes all test cases
2. **Generality**: Works for arbitrary compositions of supported operations
3. **Efficiency**: O(n) time complexity where n is number of operations
4. **Code quality**: Clear, functional q style
5. **Extensibility**: Easy to add new operations

## Deliverables

Submit a single file `AD.q` containing:
- The `AD` function
- Any helper functions needed
- All differentiation rules for required operations

The file should be loadable via `\l AD.q` and immediately usable.

## Resources

- q parse trees: Try `parse"x*x"` to see AST structure
- q function introspection: Use `value` on a function object
- Reverse-mode AD is the same as backpropagation
- Test your understanding: manually compute derivatives for simple cases first

## Getting Started

Start simple:
1. First, make `AD[{x}; 5]` return `1f` (derivative of x is 1)
2. Then handle `AD[{x*x}; 3]`
3. Gradually add more operations
4. Test frequently with the provided test cases

Good luck!
