OPENQASM 2.0;
include "qelib1.inc";
gate cs q0,q1 { p(pi/4) q0; cx q0,q1; p(-pi/4) q1; cx q0,q1; p(pi/4) q1; }
qreg q[3];
cz q[0],q[1];
u(0,0,1.5) q[2];
cp(1.5) q[0],q[1];
rz(1.5) q[2];
cs q[0],q[1];
s q[2];
cx q[1],q[2];
cz q[0],q[2];