OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
creg c[5];
rx(1.5) q[0];
ry(1.5) q[1];
cx q[2],q[3];
cx q[3],q[4];