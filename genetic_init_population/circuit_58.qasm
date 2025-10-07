OPENQASM 2.0;
include "qelib1.inc";
gate dcx q0,q1 { cx q0,q1; cx q1,q0; }
qreg q[3];
csx q[0],q[1];
u(0,0,1.5) q[2];
dcx q[0],q[1];
sxdg q[2];
cswap q[0],q[1],q[2];
cx q[1],q[2];
cz q[0],q[2];