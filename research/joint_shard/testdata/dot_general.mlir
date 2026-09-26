module {
  func.func @main(
      %x: tensor<8x16xf32>,
      %w: tensor<16x32xf32>
  ) -> tensor<8x32xf32> {
    %0 = stablehlo.dot_general %x, %w,
        contracting_dims = [1] x [0]
        : (tensor<8x16xf32>, tensor<16x32xf32>)
          -> tensor<8x32xf32>

    return %0 : tensor<8x32xf32>
  }
}
