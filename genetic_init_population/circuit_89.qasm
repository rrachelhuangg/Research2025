OPENQASM 2.0;
include "qelib1.inc";
gate ccz q0,q1,q2 { h q2; ccx q0,q1,q2; h q2; }
qreg q[3];
y q[0];
rxx(1.5) q[1],q[2];
csx q[0],q[1];
ry(1.5) q[2];
ccz q[0],q[1],q[2];
cx q[1],q[2];
cz q[0],q[2];