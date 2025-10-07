OPENQASM 2.0;
include "qelib1.inc";
gate cs q0,q1 { p(pi/4) q0; cx q0,q1; p(-pi/4) q1; cx q0,q1; p(pi/4) q1; }
gate ccz q0,q1,q2 { h q2; ccx q0,q1,q2; h q2; }
qreg q[3];
cs q[0],q[1];
u(pi/2,1.5,1.5) q[2];
ccz q[0],q[1],q[2];
ccx q[0],q[1],q[2];
cx q[1],q[2];
cz q[0],q[2];