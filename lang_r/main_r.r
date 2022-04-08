#!/usr/bin/env -S Rscript --vanilla

dyn.load("/home/rene/projekte/uni/mardi/m2/m2-mixed-experiments/cmake-build-debug/oif_connector/liboif_connector.so")
lang = "julia"
expression = "print(6*7)"
.Call("oif_connector_init_r", lang)
.Call("oif_connector_eval_expression_r", expression)
.Call("oif_connector_deinit")
