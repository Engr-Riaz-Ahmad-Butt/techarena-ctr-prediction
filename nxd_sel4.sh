#!/bin/sh
# nxd: re-select the experts' configurations on day 4 (phase 1 of --target 5: train < day 4, validate on day-4 first events).
cd /c/Users/engrr/Documents/ChatGPT/Task/techarena-ctr-prediction
B="python nxb_b_seq.py --target 5 --phase select --epochs 12 --patience 2"
$B --name nxd_sel4_w7first --no-extras --no-attn --aux 1.0 --aux-src first
$B --name nxd_sel4_w6plain --no-extras --no-attn --aux 1.0
$B --name nxd_sel4_w9attn --no-extras --aux 1.0 --aux-src first
$B --name nxd_sel4_w10neg --no-extras --no-attn --aux 1.0 --aux-src first --neg 5
$B --name nxd_sel4_w8lr --no-extras --no-attn --aux 1.0 --aux-src first --lr-emb 0.1
$B --name nxd_sel4_w4aux --no-attn --aux 1.0
$B --name nxd_sel4_w5cooc --init cooc --no-attn --aux 1.0
C="python nxb_c_mf.py --target 5 --phase select --rounds 8 --patience 2 --ledger-w 1"
$C --name nxd_sel4_anc10off --v-anchor --lv 10 --lu 1 --user-offset --lw 3
$C --name nxd_sel4_anc10 --v-anchor --lv 10 --lu 1
$C --name nxd_sel4_anc30 --v-anchor --lv 30 --lu 1
$C --name nxd_sel4_l1 --lu 1 --lv 1
$C --name nxd_sel4_rank12 --dim 12 --v-anchor --lv 10 --lu 1 --user-offset --lw 3
$C --name nxd_sel4_rank20 --dim 20 --v-anchor --lv 10 --lu 1 --user-offset --lw 3
C2="python nxb_c_mf.py --target 5 --phase select --rounds 6 --patience 2 --ledger-w 1 --v-anchor --lv 10 --lu 1 --user-offset --lw 3 --expo-w 0.1"
$C2 --name nxd_sel4_c2nk
$C2 --name nxd_sel4_c2kappa --kappa-fit
echo ALLDONE
