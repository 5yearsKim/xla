module @jit__operation attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.num_partitions = 4 : i32, mhlo.use_auto_spmd_partitioning = false} {
  sdy.mesh @mesh = <["model"=4]> {stablehlo.mesh = {axes = [{name = "model", size = 4 : i64}]}}
  func.func @main(%arg0: tensor<8x128x1024xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}]>}, %arg1: tensor<1024x1024xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"model"}]>}, %arg2: tensor<1024x1024xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"model"}]>}, %arg3: tensor<1024x1024xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"model"}]>}, %arg4: tensor<1024x1024xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"model"}, {}]>}, %arg5: tensor<1024x4096xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"model"}]>}, %arg6: tensor<4096x1024xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"model"}, {}]>}) -> (tensor<8x128x1024xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}]>}) {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x128x1024xf32>, tensor<1024x1024xf32>) -> tensor<8x128x1024xf32>
    %1 = stablehlo.dot_general %arg0, %arg2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x128x1024xf32>, tensor<1024x1024xf32>) -> tensor<8x128x1024xf32>
    %2 = stablehlo.transpose %1, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[8,1024,128]{1,2,0}"} : (tensor<8x128x1024xf32>) -> tensor<8x1024x128xf32>
    %3 = stablehlo.dot_general %0, %2, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x128x1024xf32>, tensor<8x1024x128xf32>) -> tensor<8x128x128xf32>
    %4 = sdy.constant dense<3.200000e+01> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<f32>) -> tensor<8x128x128xf32>
    %6 = stablehlo.divide %3, %5 : tensor<8x128x128xf32>
    %7 = sdy.constant dense<0xFF800000> : tensor<f32>
    %8 = stablehlo.reduce(%6 init: %7) applies stablehlo.maximum across dimensions = [2] : (tensor<8x128x128xf32>, tensor<f32>) -> tensor<8x128xf32>
    %9 = sdy.constant dense<0xFF800000> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [] : (tensor<f32>) -> tensor<8x128xf32>
    %11 = stablehlo.maximum %8, %10 : tensor<8x128xf32>
    %12 = stablehlo.reshape %11 : (tensor<8x128xf32>) -> tensor<8x128x1xf32>
    %13 = stablehlo.broadcast_in_dim %12, dims = [0, 1, 2] : (tensor<8x128x1xf32>) -> tensor<8x128x1xf32>
    %14 = stablehlo.reshape %13 : (tensor<8x128x1xf32>) -> tensor<8x128xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 1] : (tensor<8x128xf32>) -> tensor<8x128x128xf32>
    %16 = stablehlo.subtract %6, %15 : tensor<8x128x128xf32>
    %17 = stablehlo.exponential %16 : tensor<8x128x128xf32>
    %18 = sdy.constant dense<0.000000e+00> : tensor<f32>
    %19 = stablehlo.reduce(%17 init: %18) applies stablehlo.add across dimensions = [2] : (tensor<8x128x128xf32>, tensor<f32>) -> tensor<8x128xf32>
    %20 = stablehlo.reshape %19 : (tensor<8x128xf32>) -> tensor<8x128x1xf32>
    %21 = stablehlo.broadcast_in_dim %20, dims = [0, 1, 2] : (tensor<8x128x1xf32>) -> tensor<8x128x1xf32>
    %22 = stablehlo.reshape %21 : (tensor<8x128x1xf32>) -> tensor<8x128xf32>
    %23 = stablehlo.broadcast_in_dim %22, dims = [0, 1] : (tensor<8x128xf32>) -> tensor<8x128x128xf32>
    %24 = stablehlo.divide %17, %23 : tensor<8x128x128xf32>
    %25 = stablehlo.dot_general %arg0, %arg3, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x128x1024xf32>, tensor<1024x1024xf32>) -> tensor<8x128x1024xf32>
    %26 = stablehlo.dot_general %24, %25, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x128x128xf32>, tensor<8x128x1024xf32>) -> tensor<8x128x1024xf32>
    %27 = stablehlo.dot_general %26, %arg4, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x128x1024xf32>, tensor<1024x1024xf32>) -> tensor<8x128x1024xf32>
    %28 = stablehlo.add %arg0, %27 : tensor<8x128x1024xf32>
    %29 = stablehlo.dot_general %28, %arg5, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x128x1024xf32>, tensor<1024x4096xf32>) -> tensor<8x128x4096xf32>
    %30 = stablehlo.multiply %29, %29 : tensor<8x128x4096xf32>
    %31 = stablehlo.multiply %30, %29 : tensor<8x128x4096xf32>
    %32 = sdy.constant dense<4.471500e-02> : tensor<f32>
    %33 = stablehlo.broadcast_in_dim %32, dims = [] : (tensor<f32>) -> tensor<8x128x4096xf32>
    %34 = stablehlo.multiply %31, %33 : tensor<8x128x4096xf32>
    %35 = stablehlo.add %29, %34 : tensor<8x128x4096xf32>
    %36 = sdy.constant dense<0.797884583> : tensor<f32>
    %37 = stablehlo.broadcast_in_dim %36, dims = [] : (tensor<f32>) -> tensor<8x128x4096xf32>
    %38 = stablehlo.multiply %35, %37 : tensor<8x128x4096xf32>
    %39 = stablehlo.tanh %38 : tensor<8x128x4096xf32>
    %40 = sdy.constant dense<1.000000e+00> : tensor<f32>
    %41 = stablehlo.broadcast_in_dim %40, dims = [] : (tensor<f32>) -> tensor<8x128x4096xf32>
    %42 = stablehlo.add %39, %41 : tensor<8x128x4096xf32>
    %43 = sdy.constant dense<5.000000e-01> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %43, dims = [] : (tensor<f32>) -> tensor<8x128x4096xf32>
    %45 = stablehlo.multiply %42, %44 : tensor<8x128x4096xf32>
    %46 = stablehlo.multiply %29, %45 : tensor<8x128x4096xf32>
    %47 = stablehlo.dot_general %46, %arg6, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x128x4096xf32>, tensor<4096x1024xf32>) -> tensor<8x128x1024xf32>
    %48 = stablehlo.add %28, %47 : tensor<8x128x1024xf32>
    return %48 : tensor<8x128x1024xf32>
  }
}
