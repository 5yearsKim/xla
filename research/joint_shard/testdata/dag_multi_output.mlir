module {
  sdy.mesh @mesh = <["data"=2, "model"=2]>
  func.func @main(%x: tensor<8x4xf32>, %w: tensor<8x4xf32>) -> (tensor<8x4xf32>, tensor<8x4xf32>, tensor<8x4xf32>, tensor<8x4xf32>) {
    %a = stablehlo.add %x, %w : tensor<8x4xf32>
    %b = stablehlo.multiply %x, %w : tensor<8x4xf32>
    %c = stablehlo.tanh %a : tensor<8x4xf32>
    %d = stablehlo.add %c, %b : tensor<8x4xf32>
    return %d, %b, %d, %x : tensor<8x4xf32>, tensor<8x4xf32>, tensor<8x4xf32>, tensor<8x4xf32>
  }
}
