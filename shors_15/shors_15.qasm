OPENQASM 2.0;
include "qelib1.inc";
gate ccircuit_163 q0,q1,q2,q3,q4 { ccx q0,q3,q4; ccx q0,q4,q3; ccx q0,q3,q4; ccx q0,q2,q3; ccx q0,q3,q2; ccx q0,q2,q3; ccx q0,q1,q2; ccx q0,q2,q1; ccx q0,q1,q2; cx q0,q1; cx q0,q2; cx q0,q3; cx q0,q4; }
gate ccircuit_171 q0,q1,q2,q3,q4 { ccx q0,q3,q4; ccx q0,q4,q3; ccx q0,q3,q4; ccx q0,q2,q3; ccx q0,q3,q2; ccx q0,q2,q3; ccx q0,q1,q2; ccx q0,q2,q1; ccx q0,q1,q2; cx q0,q1; cx q0,q2; cx q0,q3; cx q0,q4; ccx q0,q3,q4; ccx q0,q4,q3; ccx q0,q3,q4; ccx q0,q2,q3; ccx q0,q3,q2; ccx q0,q2,q3; ccx q0,q1,q2; ccx q0,q2,q1; ccx q0,q1,q2; cx q0,q1; cx q0,q2; cx q0,q3; cx q0,q4; }
gate ccircuit_179 q0,q1,q2,q3,q4 { ccx q0,q3,q4; ccx q0,q4,q3; ccx q0,q3,q4; ccx q0,q2,q3; ccx q0,q3,q2; ccx q0,q2,q3; ccx q0,q1,q2; ccx q0,q2,q1; ccx q0,q1,q2; cx q0,q1; cx q0,q2; cx q0,q3; cx q0,q4; ccx q0,q3,q4; ccx q0,q4,q3; ccx q0,q3,q4; ccx q0,q2,q3; ccx q0,q3,q2; ccx q0,q2,q3; ccx q0,q1,q2; ccx q0,q2,q1; ccx q0,q1,q2; cx q0,q1; cx q0,q2; cx q0,q3; cx q0,q4; ccx q0,q3,q4; ccx q0,q4,q3; ccx q0,q3,q4; ccx q0,q2,q3; ccx q0,q3,q2; ccx q0,q2,q3; ccx q0,q1,q2; ccx q0,q2,q1; ccx q0,q1,q2; cx q0,q1; cx q0,q2; cx q0,q3; cx q0,q4; ccx q0,q3,q4; ccx q0,q4,q3; ccx q0,q3,q4; ccx q0,q2,q3; ccx q0,q3,q2; ccx q0,q2,q3; ccx q0,q1,q2; ccx q0,q2,q1; ccx q0,q1,q2; cx q0,q1; cx q0,q2; cx q0,q3; cx q0,q4; }
gate ccircuit_187 q0,q1,q2,q3,q4 { ccx q0,q3,q4; ccx q0,q4,q3; ccx q0,q3,q4; ccx q0,q2,q3; ccx q0,q3,q2; ccx q0,q2,q3; ccx q0,q1,q2; ccx q0,q2,q1; ccx q0,q1,q2; cx q0,q1; cx q0,q2; cx q0,q3; cx q0,q4; ccx q0,q3,q4; ccx q0,q4,q3; ccx q0,q3,q4; ccx q0,q2,q3; ccx q0,q3,q2; ccx q0,q2,q3; ccx q0,q1,q2; ccx q0,q2,q1; ccx q0,q1,q2; cx q0,q1; cx q0,q2; cx q0,q3; cx q0,q4; ccx q0,q3,q4; ccx q0,q4,q3; ccx q0,q3,q4; ccx q0,q2,q3; ccx q0,q3,q2; ccx q0,q2,q3; ccx q0,q1,q2; ccx q0,q2,q1; ccx q0,q1,q2; cx q0,q1; cx q0,q2; cx q0,q3; cx q0,q4; ccx q0,q3,q4; ccx q0,q4,q3; ccx q0,q3,q4; ccx q0,q2,q3; ccx q0,q3,q2; ccx q0,q2,q3; ccx q0,q1,q2; ccx q0,q2,q1; ccx q0,q1,q2; cx q0,q1; cx q0,q2; cx q0,q3; cx q0,q4; ccx q0,q3,q4; ccx q0,q4,q3; ccx q0,q3,q4; ccx q0,q2,q3; ccx q0,q3,q2; ccx q0,q2,q3; ccx q0,q1,q2; ccx q0,q2,q1; ccx q0,q1,q2; cx q0,q1; cx q0,q2; cx q0,q3; cx q0,q4; ccx q0,q3,q4; ccx q0,q4,q3; ccx q0,q3,q4; ccx q0,q2,q3; ccx q0,q3,q2; ccx q0,q2,q3; ccx q0,q1,q2; ccx q0,q2,q1; ccx q0,q1,q2; cx q0,q1; cx q0,q2; cx q0,q3; cx q0,q4; ccx q0,q3,q4; ccx q0,q4,q3; ccx q0,q3,q4; ccx q0,q2,q3; ccx q0,q3,q2; ccx q0,q2,q3; ccx q0,q1,q2; ccx q0,q2,q1; ccx q0,q1,q2; cx q0,q1; cx q0,q2; cx q0,q3; cx q0,q4; ccx q0,q3,q4; ccx q0,q4,q3; ccx q0,q3,q4; ccx q0,q2,q3; ccx q0,q3,q2; ccx q0,q2,q3; ccx q0,q1,q2; ccx q0,q2,q1; ccx q0,q1,q2; cx q0,q1; cx q0,q2; cx q0,q3; cx q0,q4; }
gate gate_IQFT q0,q1,q2,q3 { h q0; cp(-pi/2) q1,q0; h q1; cp(-pi/4) q2,q0; cp(-pi/2) q2,q1; h q2; cp(-pi/8) q3,q0; cp(-pi/4) q3,q1; cp(-pi/2) q3,q2; h q3; }
gate gate_IQFT_5172130592 q0,q1,q2,q3 { gate_IQFT q0,q1,q2,q3; }
qreg q[8];
h q[0];
h q[1];
h q[2];
h q[3];
x q[7];
ccircuit_163 q[0],q[4],q[5],q[6],q[7];
ccircuit_171 q[1],q[4],q[5],q[6],q[7];
ccircuit_179 q[2],q[4],q[5],q[6],q[7];
ccircuit_187 q[3],q[4],q[5],q[6],q[7];
gate_IQFT_5172130592 q[0],q[1],q[2],q[3];