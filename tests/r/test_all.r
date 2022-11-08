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
assert(is.loaded("oif_connector_init_r"))

ret = .Call("oif_connector_init_r", lang)
assert(ret==0)
ret = .Call("oif_connector_eval_expression_r", expr)
assert(ret==0)
ret = .Call("oif_connector_deinit")
assert(ret==0)
