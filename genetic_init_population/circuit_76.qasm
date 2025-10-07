OPENQASM 2.0;
include "qelib1.inc";
gate csdg q0,q1 { p(-pi/4) q0; cx q0,q1; p(pi/4) q1; cx q0,q1; p(-pi/4) q1; }
qreg q[3];
u(1.5,2.5,3.5) q[0];
sxdg q[1];
rz(1.5) q[2];
h q[0];
y q[1];
tdg q[2];
csdg q[0],q[1];
rx(1.5) q[2];
cx q[1],q[2];
cz q[0],q[2];