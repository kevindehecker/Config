target extended-remote /dev/ttyACM0
monitor tpwr enable
monitor swdp_scan
attach 1
monitor vector_catch disable hard
set mem inaccessible-by-default off
set print pretty