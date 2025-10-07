OPENQASM 2.0;
include "qelib1.inc";
gate ccz q0,q1,q2 { h q2; ccx q0,q1,q2; h q2; }
gate dcx q0,q1 { cx q0,q1; cx q1,q0; }
qreg q[3];
ccz q[0],q[1],q[2];
y q[0];
cry(1.5) q[1],q[2];
dcx q[0],q[1];
tdg q[2];
cx q[1],q[2];
cz q[0],q[2];