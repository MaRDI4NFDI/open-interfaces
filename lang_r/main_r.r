#!/usr/bin/env Rscript

dyn.load("liboif_connector.so")
.C("oif_connector_init", "python")
.C("oif_connector_eval_expression", "print(6*7)")
.C("oif_connector_eval_expression")
