OPENQASM 2.0;
include "qelib1.inc";
gate ccz q0,q1,q2 { h q2; ccx q0,q1,q2; h q2; }
qreg q[3];
id q[0];
cz q[1],q[2];
u(pi/2,1.5,1.5) q[0];
h q[1];
p(1.5) q[2];
ccz q[0],q[1],q[2];
cx q[1],q[2];
cz q[0],q[2];