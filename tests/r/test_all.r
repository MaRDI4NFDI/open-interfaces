#!/usr/bin/env -S Rscript --vanilla
library(testit)

args = commandArgs(trailingOnly=TRUE)
if (length(args)!=2) {
  lang = "julia"
  expr = "print(6*7)"
} else {
  lang = args[1]
  expr = args[2]
}

connector_path = Sys.getenv("R_LIBOIF_CONNECTOR")
dyn.load(connector_path)
.Call("oif_connector_init_r", lang)
assert(is.loaded("oif_connector_init_r"))
# .Call("oif_connector_eval_expression_r", expr)
# .Call("oif_connector_deinit")
