#include <iostream>

#include "llvm/Support/raw_ostream.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "stablehlo/dialect/Register.h"
#include "stablehlo/dialect/StablehloOps.h"

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: parse_stablehlo <file.mlir>\n";
        return 1;
    }

    mlir::DialectRegistry registry;

    // Registers StableHLO and the dialects needed by StableHLO.
    mlir::stablehlo::registerAllDialects(registry);

    mlir::MLIRContext context(registry);

    auto module =
        mlir::parseSourceFile<mlir::ModuleOp>(argv[1], &context);

    if (!module) {
        std::cerr << "Failed to parse MLIR file\n";
        return 1;
    }

    llvm::outs() << "Successfully parsed StableHLO\n\n";

    module->walk([](mlir::Operation* op) {
        llvm::outs()
            << "op: "
            << op->getName().getStringRef()
            << "\n";

        if (auto dot =
                mlir::dyn_cast<mlir::stablehlo::DotGeneralOp>(op)) {
            llvm::outs()
                << "  -> Found typed stablehlo::DotGeneralOp\n";

            llvm::outs()
                << "  -> LHS type: "
                << dot.getLhs().getType()
                << "\n";

            llvm::outs()
                << "  -> RHS type: "
                << dot.getRhs().getType()
                << "\n";

            llvm::outs()
                << "  -> dot dimension numbers: "
                << dot.getDotDimensionNumbers()
                << "\n";
        }
    });

    llvm::outs() << "\nFull module:\n";
    module->print(llvm::outs());
    llvm::outs() << "\n";

    return 0;
}
