# Contributing to TRTInferX

Thank you for your interest in contributing to **TRTInferX**.  
This project focuses on high-performance TensorRT inference for YOLOv11 in
real-world robotics and radar applications.

Contributions are welcome in the form of code, documentation, bug reports,
and performance evaluations.

---

## Scope of Contributions

We particularly welcome contributions related to:

- TensorRT inference optimization (FP16 / INT8)
- CUDA preprocessing or postprocessing kernels
- Performance benchmarking and analysis
- Documentation improvements or clarifications
- Bug fixes and stability improvements

Major architectural changes should be discussed first.

---

## How to Contribute

1. **Fork** the repository and create a new branch from `main`.
2. Make your changes with clear and minimal commits.
3. Ensure the code builds successfully on your target platform.
4. Submit a **Pull Request** with a concise description of:
   - What was changed
   - Why the change is needed
   - Any performance or behavior impact

---

## Code Style and Guidelines

- C++17 is required.
- Keep changes focused and avoid unrelated refactoring.
- Follow existing coding style and file structure.
- CUDA code should prioritize correctness and clarity over micro-optimizations,
  unless performance gains are clearly demonstrated.

---

## Testing and Validation

If your change affects inference behavior or performance:

- Provide basic validation steps or test commands.
- Include performance numbers when applicable
  (hardware, batch size, precision, TensorRT version).

---

## Licensing

By contributing to this project, you agree that your contributions will be
licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**,
consistent with the rest of the project.

---

## Communication

If you are unsure whether a change fits the project scope, please open an
Issue for discussion before submitting a Pull Request.

Thank you for helping improve TRTInferX.
