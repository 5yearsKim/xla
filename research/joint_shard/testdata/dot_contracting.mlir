module {
  sdy.mesh @mesh = <["data"=2, "model"=2]>
  func.func @main(
      %x: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"data"}, {"model"}]>},
      %w: tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"model"}, {}]>}
  ) -> (tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"data"}, {}]>}) {
    %0 = stablehlo.dot_general %x, %w,
        contracting_dims = [1] x [0]
        : (tensor<8x16xf32>, tensor<16x32xf32>)
          -> tensor<8x32xf32>

    return %0 : tensor<8x32xf32>
  }
}
