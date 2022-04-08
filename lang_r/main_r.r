#!/usr/bin/env -S Rscript --vanilla

args = commandArgs(trailingOnly=TRUE)
if (length(args)!=2) {
  lang = "julia"
  expr = "print(6*7)"
} else {
  lang = args[1]
  expr = args[2]
}

dyn.load("/home/rene/projekte/uni/mardi/m2/m2-mixed-experiments/cmake-build-debug/oif_connector/liboif_connector.so")
.Call("oif_connector_init_r", lang)
.Call("oif_connector_eval_expression_r", expr)
.Call("oif_connector_deinit")
