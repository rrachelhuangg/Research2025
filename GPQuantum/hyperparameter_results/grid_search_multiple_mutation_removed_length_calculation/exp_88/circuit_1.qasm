OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
creg c[5];
rx(1.5) q[0];
rz(1.5) q[1];
ry(1.5) q[2];
cx q[3],q[4];