import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT
from qiskit.circuit.library.standard_gates import TGate
from qiskit.extensions import UnitaryGate


class IllegalArgumentError(ValueError):
    pass


class QuantumPhaseEstimation:
    @classmethod
    def get_circuit(
        cls,
        counting_qubits: int,
        ancillary_qubits: int,
        unitary_matrix: np.ndarray = None,
        phase_angle: float = None,
    ) -> QuantumCircuit:

        if not ((phase_angle is None) ^ (unitary_matrix is None)):
            raise IllegalArgumentError(
                "exactly one of phase or a unitary matrix must be provided"
            )

        cq, aq = counting_qubits, ancillary_qubits
        nq = counting_qubits + ancillary_qubits

        circuit = QuantumCircuit(nq, cq)

        circuit = cls._initialize_system(circuit, cq, aq, unitary_matrix, phase_angle)

        circuit.barrier()
        circuit = circuit.compose(QFT(cq, inverse=True), range(cq))

        circuit.measure(range(cq), range(cq))

        return circuit

    @staticmethod
    def _initialize_system(
        circuit: QuantumCircuit,
        cq: int,
        aq: int,
        unitary_matrix: np.ndarray,
        phase_angle: float,
    ) -> QuantumCircuit:

        circuit.h(range(cq))
        circuit.x(range(cq, cq + aq))

        uc = (
            UnitaryGate(unitary_matrix, label="U").control(aq)
            if phase_angle is None
            else False
        )

        if aq > 1 and uc is False:
            ctgate = TGate().control(aq)
            uc = ctgate

        for x in reversed(range(cq)):
            for _ in range(2 ** (cq - 1 - x)):
                if uc is not False:
                    circuit.append(uc, [cq - 1 - x, *list(range(cq, cq + aq))])
                else:
                    circuit.cp(phase_angle, cq - 1 - x, cq)

        return circuit
