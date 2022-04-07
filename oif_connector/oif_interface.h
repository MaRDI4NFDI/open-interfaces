//
// Created by rene on 07.04.22.
//

#ifndef OIF_EXPERIMENTS_OIF_INTERFACE_H
#define OIF_EXPERIMENTS_OIF_INTERFACE_H

int oif_lang_init();
int oif_lang_eval_expression(const char *str);
void oif_lang_deinit();

#endif // OIF_EXPERIMENTS_OIF_INTERFACE_H
