# TorchSig Dataset Builder

### Functional Specification & Project Plan

---

## 1. Overview

The TorchSig Dataset Builder is a graphical utility application for designing, configuring, and generating synthetic RF signal datasets using TorchSig.

The system emphasizes:

* Rapid dataset creation without manual coding
* Interactive, context-aware configuration
* Reproducibility via YAML-based macros
* Validation of signal generation and transform pipelines

This application serves as a **dataset design and orchestration layer**, not a replacement for TorchSig or RFML tooling.

---

## 2. Core Design Principles

* **GUI-first, not config-first**
* **Metadata-driven validation (no hardcoded logic)**
* **Global defaults with explicit overrides**
* **Composable pipelines**
* **Reproducibility as a first-class feature**

---

## 3. System Architecture

### 3.1 High-Level Flow

1. User selects signal generators
2. System loads global parameter defaults
3. User optionally applies per-generator overrides
4. User constructs transform pipeline
5. System validates compatibility
6. User configures dataset output
7. User optionally loads/saves macros
8. Dataset generation executes

---

### 3.2 Core Modules

* Generator Registry
* Parameter Manager (global + overrides)
* Transform Pipeline Engine
* Compatibility Validator
* Dataset Generator
* Macro Manager (YAML I/O)
* GUI Layer

---

## 4. Signal Generator Configuration

### 4.1 Generator Selection

* Multi-select interface (checkbox list)
* Dynamically updates available parameters

---

### 4.2 Parameter System

#### Global Parameters ("Blanket")

Apply to all applicable generators and transforms:

* Sample rate
* Duration
* SNR range
* Bandwidth
* Other shared properties

---

#### Per-Generator Overrides

* Default: inherit global values
* Editable per generator
* Override persists even if global value changes
* Option to reset to global defaults

---

### 4.3 Specialized Generators (e.g., LFM)

* Selecting a generator injects required parameter groups into the UI
* Parameters are conditionally rendered
* Marked as required for validation

---

## 5. Transform Pipeline

### 5.1 Transform Library

* Selectable transform sets (e.g., impairments, augmentations)

---

### 5.2 Pipeline Construction

* Add/remove transforms
* Reorder via drag-and-drop
* Enable/disable individual transforms

---

### 5.3 Transform Scope

* Apply to:

  * Entire dataset
  * Subsets of samples (future support)

---

## 6. Compatibility & Validation System

### 6.1 Metadata-Driven Compatibility

Each generator and transform defines:

```yaml
generator:
  name: ExampleGenerator
  produces: [complex_iq]
  requires: [baseband]

transform:
  name: ExampleTransform
  accepts: [complex_iq]
  modifies: [snr]
```

---

### 6.2 Validation Rules

A pipeline is valid if:

1. **Type Compatibility**

   * Output of generator matches input of first transform
   * Each transform output matches next transform input

2. **Parameter Completeness**

   * All required parameter groups are defined

3. **Constraint Satisfaction**

   * No explicit incompatibilities are violated

---

### 6.3 Constraint System

Transforms may define:

```yaml
constraints:
  incompatible_with: [chirp_preserving]
```

Generators may define semantic tags:

* `chirp_preserving`
* `wideband`
* `narrowband`

---

### 6.4 UI Behavior

* Real-time validation during configuration
* Visual indicators:

  * ⚠️ Warning (potential issue)
  * ❌ Error (invalid configuration)

---

### 6.5 Selection Strategy

**Allow but warn (default behavior):**

* Users can select incompatible combinations
* System highlights issues
* Dataset generation is blocked until resolved

---

### 6.6 Auto-Resolution

#### Suggested Fixes

* Recommend compatible transforms
* Suggest missing parameters
* Propose adapter transforms

---

#### Adapter Transforms

Built-in bridging transforms:

* Complex → Real (magnitude, phase)
* Resampling
* Bandwidth normalization

---

### 6.7 Macro Validation

When loading macros:

* Validate entire configuration
* Highlight issues
* Allow partial loading with warnings

---

### 6.8 Runtime Safety

* Hard validation before dataset generation
* Clear error messages with:

  * Pipeline location
  * Cause of failure
  * Suggested fixes

---

## 7. Dataset Generation

### 7.1 User Inputs

* Total sample count
* Output format selector:
  * TorchSig-compatible HDF5
  * NumPy split folders
* Train/validation split ratio for NumPy output

---

### 7.2 Output Structure

* TorchSig-compatible HDF5 output produces a `data.h5` dataset at the root, plus `config.yaml` and small metadata files alongside it.
* NumPy output produces the existing split-folder layout:

```text
dataset/
├── train/
│   ├── raw/
│   └── impaired/
├── val/
│   ├── raw/
│   └── impaired/
```

---

### 7.3 Definitions

* **Raw**: Untransformed signals
* **Impaired**: Signals after transform pipeline

---

### 7.4 Execution Flow

1. Generate the selected signal representation
2. Apply the transform pipeline
3. Write either TorchSig-compatible HDF5 or NumPy split folders
4. Package the output for download

---

## 8. Macro System (YAML Configurations)

### 8.1 Features

* Load macros from YAML
* Save current configuration as macro

---

### 8.2 Macro Contents

* Generator selection
* Global parameters
* Per-generator overrides
* Transform pipeline
* Dataset configuration

---

### 8.3 Default Macro

#### Sig53 Replica

* Matches TorchSig Sig53 dataset:

  * Concrete TorchSig generator roster
  * Uniform class distribution across the full class list
  * Wideband multi-signal recipe via `num_signals_min=3` and `num_signals_max=5`
  * Parameter distributions and transform pipeline aligned with the current TorchSig 2.1-style metadata syntax
  * Dataset structure and metadata written alongside the exported data

---

### 8.4 Versioning

* Schema version included in YAML
* Backward compatibility handling

---

## 9. User Interface Design

### 9.1 Layout Sections

* Generator Selection Panel
* Parameter Configuration Panel
* Transform Pipeline Panel
* Dataset Output Panel
* Macro Management Panel

---

### 9.2 Dynamic Behavior

* Context-aware parameter rendering
* Visual indicators for overrides
* Inline validation feedback

---

## 10. Data Flow

1. Select generators
2. Apply global parameters
3. Override as needed
4. Build transform pipeline
5. Validate configuration
6. Configure dataset output
7. Load/save macro (optional)
8. Generate dataset

---

## 11. Extensibility

* Plugin-based generator registry
* Pluggable transform libraries
* Schema-driven compatibility system
* CLI/API compatibility (future)

---

## 12. Future Enhancements

* Signal preview (pre-generation)
* Pipeline simulation mode
* Compatibility scoring
* Auto-pipeline generation (goal-driven)
* Distributed dataset generation

---

## 13. Positioning

This application is:

* A **dataset design tool**
* A **visual orchestration layer** for TorchSig

This application is not:

* A replacement for TorchSig
* A full RFML training framework

---

## 14. Key Risks

* State management complexity (global vs overrides)
* UI complexity for dynamic parameter rendering
* Maintaining compatibility metadata accuracy

---

## 15. Success Criteria

* Users can generate valid datasets without writing code
* Configuration errors are caught before runtime
* Macros enable reproducible dataset generation
* Time-to-dataset is significantly reduced compared to manual workflows

---


---

## 16. Initial Implementation (Current Repo State)

A first runnable scaffold is now included:

- `app.py`: Streamlit GUI for generator selection, global params, transforms, macro load/save, validation, and dataset generation.
- `sigsynth/`: Core modules for metadata registry, validation, macro I/O, and placeholder generation.
- `macros/sig53.yaml`: Starter macro for Sig53-style narrowband setup.
- `macros/wideband_sig53.yaml`: Starter macro for wideband Sig53-style setup.

### Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

### Run locally with uv (preferred)

```bash
# initialize torchsig submodule (once after clone)
git submodule update --init --recursive

# create/update environment with local torchsig extra enabled
uv sync --extra torchsig-local
uv run --extra torchsig-local streamlit run app.py
```

### Streamlit Cloud

Use the base app install only.

- Do not enable `torchsig-local` in the cloud deployment.
- Do not rely on the `third_party/torchsig` submodule being built in Streamlit Cloud.
- The app will run with the NumPy or local `h5py` HDF5 fallback there.
- `uv.lock` is generated from `pyproject.toml` and the vendored TorchSig metadata, so manual edits to the lockfile can be overwritten the next time the lock is refreshed.

### Notes

- The app now lets you choose between TorchSig-compatible HDF5 output and NumPy split folders. For HDF5, it uses TorchSig-native generation when available and falls back to a local `h5py` writer with the same `data.h5` layout when TorchSig is unavailable.
- Legacy or unknown generator and transform names stay visible in the UI so they can be remapped instead of silently disappearing from loaded macros.
- The UI shows whether the runtime is using TorchSig-backed generation or the local HDF5 fallback, and the transform preview is labeled as approximate because it uses a synthetic sample for fast feedback.
- Generated datasets are sandboxed under `output/`, and macro files are constrained to the `macros/` directory.
- After generation, the app provides a dataset ZIP download button so hosted Streamlit deployments can export generated artifacts.
