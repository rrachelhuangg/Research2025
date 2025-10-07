OPENQASM 2.0;
include "qelib1.inc";
gate cs q0,q1 { p(pi/4) q0; cx q0,q1; p(-pi/4) q1; cx q0,q1; p(pi/4) q1; }
qreg q[3];
cs q[0],q[1];
p(1.5) q[2];
x q[0];
cy q[1],q[2];
t q[0];
tdg q[1];
t q[2];
cx q[1],q[2];
cz q[0],q[2];