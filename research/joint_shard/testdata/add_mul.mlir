module {
  func.func @main(
      %a: tensor<2x2xf32>,
      %b: tensor<2x2xf32>,
      %c: tensor<2x2xf32>
  ) -> tensor<2x2xf32> {
    %0 = stablehlo.multiply %a, %b : tensor<2x2xf32>
    %1 = stablehlo.add %0, %c : tensor<2x2xf32>
    return %1 : tensor<2x2xf32>
  }
}
