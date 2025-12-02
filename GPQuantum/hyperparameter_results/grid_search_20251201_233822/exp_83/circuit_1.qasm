OPENQASM 2.0;
include "qelib1.inc";
qreg q[6];
creg c[6];
ry(1.5) q[0];
cx q[1],q[2];
cx q[2],q[3];
ry(1.5) q[3];