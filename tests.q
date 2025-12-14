/ Unit tests for automatic differentiation
/ Load the AD module
system"l AD.q";

/ Assert function - raises error if condition is false
assert:{[bool; err]
  if[not bool; '`$err]
  };

/ Helper to check approximate equality for floats
assertClose:{[a; b; err]
  if[not (abs[a-b]) < 0.00001; '`$err, " (expected: ", string[b], ", got: ", string[a], ")"]
  };

runTests:{
  show "=== Testing Automatic Differentiation ===";
  
  / Basic test: identity function
  show "Test 1: Identity function";
  assertClose[AD[{x}; 5]; 1f; "AD[{x}; 5] should be 1"];
  
  / Test 2: x*x (power of 2)
  show "Test 2: x*x";
  assertClose[AD[{x*x}; 3]; 6f; "AD[{x*x}; 3] should be 6"];
  
  / Test 3: x+x
  show "Test 3: x+x";
  assertClose[AD[{x+x}; 5]; 2f; "AD[{x+x}; 5] should be 2"];
  
  / Test 4: x-x
  show "Test 4: x-x";
  assertClose[AD[{x-x}; 3]; 0f; "AD[{x-x}; 3] should be 0"];
  
  / Test 5: x%x
  show "Test 5: x%x";
  assertClose[AD[{x%x}; 3]; 0f; "AD[{x%x}; 3] should be 0"];
  
  / Composed operations (q is right-to-left!)
  show "Test 6: (x*x)-x";
  assertClose[AD[{(x*x)-x}; 3]; 5f; "AD[{(x*x)-x}; 3] should be 5"];
  
  show "Test 7: (x+x)*(x+x)";
  assertClose[AD[{(x+x)*(x+x)}; 3]; 24f; "AD[{(x+x)*(x+x)}; 3] should be 24"];
  
  show "Test 8: x%(x+x)";
  assertClose[AD[{x%(x+x)}; 3]; 0f; "AD[{x%(x+x)}; 3] should be 0"];
  
  / Power functions
  show "Test 9: xexp[x;2]";
  assertClose[AD[{xexp[x;2]}; 3]; 6f; "AD[{xexp[x;2]}; 3] should be 6"];
  
  show "Test 10: xexp[x;3]";
  assertClose[AD[{xexp[x;3]}; 2]; 12f; "AD[{xexp[x;3]}; 2] should be 12"];
  
  show "Test 11: xexp[2;x]";
  assertClose[AD[{xexp[2;x]}; 3]; 5.545177; "AD[{xexp[2;x]}; 3] should be 5.545177"];
  
  / Unary functions
  show "Test 12: exp x";
  assertClose[AD[{exp x}; 0]; 1f; "AD[{exp x}; 0] should be 1"];
  
  show "Test 13: log x";
  assertClose[AD[{log x}; 2]; 0.5; "AD[{log x}; 2] should be 0.5"];
  
  show "Test 14: neg x";
  assertClose[AD[{neg x}; 3]; -1f; "AD[{neg x}; 3] should be -1"];
  
  / Complex compositions
  show "Test 15: exp x*x";
  assertClose[AD[{exp x*x}; 1]; 5.436564; "AD[{exp x*x}; 1] should be 5.436564"];
  
  show "Test 16: log xexp[x;2]";
  assertClose[AD[{log xexp[x;2]}; 2]; 1f; "AD[{log xexp[x;2]}; 2] should be 1"];
  
  show "Test 17: (x*x)%x";
  assertClose[AD[{(x*x)%x}; 3]; 1f; "AD[{(x*x)%x}; 3] should be 1"];
  
  / Partial evaluation (projection)
  show "Test 18: Partial evaluation";
  grad_f: AD[{x*x}];
  assertClose[grad_f[3]; 6f; "grad_f[3] should be 6"];
  results: grad_f each 1 2 3 4;
  assert[(results~2 4 6 8f); "grad_f each 1 2 3 4 should be 2 4 6 8"];
  
  show "All tests passed!";
  };

/ Execute with error trapping - shows backtrace on failure
.Q.trp[runTests; ::; {-1 "ERROR: ", x, "\nbacktrace:\n", .Q.sbt y; exit 1}];
exit 0  / Exit cleanly
