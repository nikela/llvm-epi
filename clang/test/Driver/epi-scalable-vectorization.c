// RUN: %clang %s -mepi -### 2>&1 | FileCheck --check-prefix=MEPI %s
// RUN: %clang %s -mepi -mepi-allow-fixed-vectorization -### 2>&1 \
// RUN:   | FileCheck --check-prefix=MEPI-FIXED %s

// MEPI:  "--scalable-vectorization=only
// MEPI-FIXED-NOT:  "--scalable-vectorization=
