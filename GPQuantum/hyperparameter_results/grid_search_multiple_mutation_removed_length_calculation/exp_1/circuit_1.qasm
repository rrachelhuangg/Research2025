OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
creg c[5];
cx q[0],q[1];
ry(1.5) q[1];
rx(1.5) q[2];