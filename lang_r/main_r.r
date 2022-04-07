#!/usr/bin/env -S Rscript --vanilla

dyn.load("/home/rene/projekte/uni/mardi/m2/m2-mixed-experiments/cmake-build-debug/oif_connector/liboif_connector.so")
.C("oif_connector_eval_expression", "print(6*7)")
.C("oif_connector_eval_expression")
