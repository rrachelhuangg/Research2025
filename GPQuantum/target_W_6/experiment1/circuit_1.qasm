OPENQASM 2.0;
include "qelib1.inc";
qreg q[6];
creg c[6];
rx(1.5) q[0];
rz(1.5) q[3];
rx(1.5) q[4];
ry(1.5) q[4];