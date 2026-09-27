module {
  sdy.mesh @mesh = <["data"=2, "model"=2]>
  func.func @main(%s: tensor<f32>, %x: tensor<8x16xf32>, %w: tensor<16x4xf32>, %v: tensor<4x4xf32>) -> tensor<8x4xf32> {
    %scale = stablehlo.broadcast_in_dim %s, dims = [] : (tensor<f32>) -> tensor<8x16xf32>
    %scaled = stablehlo.multiply %scale, %x : tensor<8x16xf32>
    %a = stablehlo.dot_general %scaled, %w, contracting_dims = [1] x [0] : (tensor<8x16xf32>, tensor<16x4xf32>) -> tensor<8x4xf32>
    %b = stablehlo.tanh %a : tensor<8x4xf32>
    %c = stablehlo.dot_general %b, %v, contracting_dims = [1] x [0] : (tensor<8x4xf32>, tensor<4x4xf32>) -> tensor<8x4xf32>
    %y = stablehlo.add %a, %c : tensor<8x4xf32>
    return %y : tensor<8x4xf32>
  }
}
