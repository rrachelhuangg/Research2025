OPENQASM 2.0;
include "qelib1.inc";
gate iswap q0,q1 { s q0; s q1; h q0; cx q0,q1; cx q1,q0; h q1; }
qreg q[3];
cu(1.5,1.5,1.5,1.5) q[0],q[1];
id q[2];
iswap q[0],q[1];
sxdg q[2];
tdg q[0];
u(1.5,2.5,3.5) q[1];
x q[2];
cx q[1],q[2];
cz q[0],q[2];