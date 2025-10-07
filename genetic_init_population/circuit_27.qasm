OPENQASM 2.0;
include "qelib1.inc";
gate ccz q0,q1,q2 { h q2; ccx q0,q1,q2; h q2; }
gate csdg q0,q1 { p(-pi/4) q0; cx q0,q1; p(pi/4) q1; cx q0,q1; p(-pi/4) q1; }
qreg q[3];
cswap q[0],q[1],q[2];
ccz q[0],q[1],q[2];
tdg q[0];
csdg q[1],q[2];
cx q[1],q[2];
cz q[0],q[2];