module {
  sdy.mesh @mesh = <["data"=2, "model"=2]>
  func.func @main(%s: tensor<f32>, %a: tensor<8x16xf32>, %b: tensor<16x4xf32>, %w: tensor<4x4xf32>) -> tensor<8x4xf32> {
    %scale = stablehlo.broadcast_in_dim %s, dims = [] : (tensor<f32>) -> tensor<8x16xf32>
    %scaled = stablehlo.multiply %scale, %a : tensor<8x16xf32>
    %dot = stablehlo.dot_general %scaled, %b, contracting_dims = [1] x [0] : (tensor<8x16xf32>, tensor<16x4xf32>) -> tensor<8x4xf32>
    %activated = stablehlo.tanh %dot : tensor<8x4xf32>
    %second = stablehlo.dot_general %activated, %w, contracting_dims = [1] x [0] : (tensor<8x4xf32>, tensor<4x4xf32>) -> tensor<8x4xf32>
    %activated2 = stablehlo.tanh %second : tensor<8x4xf32>
    return %activated2 : tensor<8x4xf32>
  }
}
