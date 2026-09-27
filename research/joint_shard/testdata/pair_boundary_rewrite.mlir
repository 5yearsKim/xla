// RHS tensors are large while dot outputs are small. Factoring saves compute
// with replicated inputs, but conflicting RHS layouts can make it communicate
// a large tensor instead of small dot outputs.
module {
  sdy.mesh @mesh = <["data"=2, "model"=2]>
  func.func @main(%a: tensor<2x256xf32>, %b: tensor<256x256xf32>, %c: tensor<256x256xf32>) -> tensor<2x256xf32> {
    %ab = stablehlo.dot_general %a, %b, contracting_dims = [1] x [0]
      : (tensor<2x256xf32>, tensor<256x256xf32>) -> tensor<2x256xf32>
    %ac = stablehlo.dot_general %a, %c, contracting_dims = [1] x [0]
      : (tensor<2x256xf32>, tensor<256x256xf32>) -> tensor<2x256xf32>
    %sum = stablehlo.add %ab, %ac : tensor<2x256xf32>
    %out = stablehlo.tanh %sum : tensor<2x256xf32>
    return %out : tensor<2x256xf32>
  }
}
