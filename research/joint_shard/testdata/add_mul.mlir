module {
  sdy.mesh @mesh = <["data"=2, "model"=2]>
  func.func @main(
      %a: tensor<2x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"data"}, {}]>},
      %b: tensor<2x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"data"}, {}]>},
      %c: tensor<2x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"data"}, {}]>}
  ) -> (tensor<2x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"data"}, {}]>}) {
    %0 = stablehlo.multiply %a, %b : tensor<2x2xf32>
    %1 = stablehlo.add %0, %c : tensor<2x2xf32>
    return %1 : tensor<2x2xf32>
  }
}
