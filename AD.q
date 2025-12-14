/ Automatic Differentiation in q/kdb+
/ Implements reverse-mode automatic differentiation

/ Differentiation rules for binary operations
/ Each rule takes (grad_output, left_value, right_value) and returns (grad_left, grad_right)
binaryRules:()!();
binaryRules[*]:{[g;l;r] (g*r; g*l)};  / Product rule: d/dx(a*b) = b, d/dy(a*b) = a
binaryRules[+]:{[g;l;r] (g; g)};      / Addition: d/dx(a+b) = 1, d/dy(a+b) = 1
binaryRules[-]:{[g;l;r] (g; neg g)};  / Subtraction: d/dx(a-b) = 1, d/dy(a-b) = -1
binaryRules[%]:{[g;l;r] (g%r; neg g*l%r*r)};  / Division: d/dx(a/b) = 1/b, d/dy(a/b) = -a/b²
binaryRules[xexp]:{[g;l;r] (g*r*xexp[l;r-1]; g*xexp[l;r]*log l)};  / Power: d/dx(a^b) = b*a^(b-1), d/dy(a^b) = a^b*ln(a)

/ Differentiation rules for unary operations
/ Each rule takes (grad_output, input_value) and returns grad_input
unaryRules:()!();
unaryRules[exp]:{[g;v] g*exp v};     / d/dx(e^x) = e^x
unaryRules[log]:{[g;v] g%v};         / d/dx(ln(x)) = 1/x
unaryRules[-:]:{[g;v] neg g};        / d/dx(-x) = -1 (neg is -: in AST)

/ Helper function to build node for an AST element
buildNode:{[ast; x; nodes]
  / Variable: return node 0 (input)
  if[-11h = type ast;
    if[ast~`x; :enlist[nodes; 0]];
    '"Unknown variable"
  ];
  
  / Constant: create constant node
  if[(type ast) within -9 -5;
    idx: count nodes;
    node: `type`value`op`parents`leftVal`rightVal!(
      `const; ast; ::; `int$(); 0n; 0n);
    nodes: nodes, enlist node;
    :enlist[nodes; idx]
  ];
  
  / Unary operation
  if[2 = count ast;
    op: first ast;
    operand: ast 1;
    
    / Recursively build operand node
    res: buildNode[operand; x; nodes];
    nodes: res 0; operandIdx: res 1;
    
    / Compute value
    operandVal: (nodes[operandIdx])`value;
    val: op[operandVal];
    
    / Create operation node
    idx: count nodes;
    node: `type`value`op`parents`leftVal`rightVal!(
      `unop; val; op; enlist operandIdx; operandVal; 0n);
    nodes: nodes, enlist node;
    :enlist[nodes; idx]
  ];
  
  / Binary operation
  if[3 = count ast;
    op: first ast;
    left: ast 1;
    right: ast 2;
    
    / Recursively build left node
    res: buildNode[left; x; nodes];
    nodes: res 0; leftIdx: res 1;
    
    / Recursively build right node
    res: buildNode[right; x; nodes];
    nodes: res 0; rightIdx: res 1;
    
    / Compute value
    leftVal: (nodes[leftIdx])`value;
    rightVal: (nodes[rightIdx])`value;
    val: op[leftVal; rightVal];
    
    / Create operation node
    idx: count nodes;
    node: `type`value`op`parents`leftVal`rightVal!(
      `binop; val; op; (leftIdx; rightIdx); leftVal; rightVal);
    nodes: nodes, enlist node;
    :enlist[nodes; idx]
  ];
  
  '"Unhandled AST structure"
  };

/ Forward pass: build computation graph
forward:{[ast; x]
  / Node 0 is always input variable
  nodes: enlist `type`value`op`parents`leftVal`rightVal!(
    `var; x; ::; `int$(); 0n; 0n);
  res: buildNode[ast; x; nodes];
  res 0
  };

/ Backward pass: compute gradients
backward:{[nodes]
  n: count nodes;
  grads: n#0f;  / Initialize all gradients to 0
  grads[n-1]: 1f;  / Output gradient is 1
  
  / Traverse nodes backward
  i: n-1;
  while[i >= 0;
    node: nodes[i];
    grad: grads[i];
    
    / If this is a unary operation, propagate gradient
    if[node[`type] = `unop;
      op: node[`op];
      parents: node[`parents];
      operandIdx: first parents;
      operandVal: node[`leftVal];  / We stored operand value in leftVal
      
      / Get gradient rule for this operation
      rule: unaryRules[op];
      localGrad: rule[grad; operandVal];
      
      / Accumulate gradient
      grads[operandIdx]: grads[operandIdx] + localGrad;
    ];
    
    / If this is a binary operation, propagate gradient
    if[node[`type] = `binop;
      op: node[`op];
      parents: node[`parents];
      leftIdx: parents 0;
      rightIdx: parents 1;
      leftVal: node[`leftVal];
      rightVal: node[`rightVal];
      
      / Get gradient rule for this operation
      rule: binaryRules[op];
      localGrads: rule[grad; leftVal; rightVal];
      
      / Accumulate gradients (important for x*x where x appears twice)
      grads[leftIdx]: grads[leftIdx] + localGrads 0;
      grads[rightIdx]: grads[rightIdx] + localGrads 1;
    ];
    
    i: i - 1;
  ];
  
  grads
  };

/ Main AD function
AD:{[f;x]
  / Extract function body and parse it
  source: last value f;  / Get source code (last element)
  body: 1 _ -1 _ source;  / Remove { and }
  ast: parse body;  / Parse to AST
  
  / Special case: just variable
  if[ast~`x; :1f];
  
  / Forward pass
  nodes: forward[ast; x];
  
  / Backward pass
  grads: backward[nodes];
  
  / Return gradient of input (first node should be the variable)
  grads[0]
  };
