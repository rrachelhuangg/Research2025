OPENQASM 2.0;
include "qelib1.inc";
gate ccz q0,q1,q2 { h q2; ccx q0,q1,q2; h q2; }
gate iswap q0,q1 { s q0; s q1; h q0; cx q0,q1; cx q1,q0; h q1; }
qreg q[3];
ccz q[0],q[1],q[2];
iswap q[0],q[1];
s q[2];
ccx q[0],q[1],q[2];
cx q[1],q[2];
cz q[0],q[2];