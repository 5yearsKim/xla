module {
  sdy.mesh @mesh = <["data"=2, "model"=2]>
  func.func @main(%x: tensor<8x4xf32>, %w: tensor<8x4xf32>) -> tensor<8x4xf32> {
    %a = stablehlo.tanh %x : tensor<8x4xf32>
    %b = stablehlo.tanh %w : tensor<8x4xf32>
    %y = stablehlo.add %a, %b : tensor<8x4xf32>
    return %y : tensor<8x4xf32>
  }
}
