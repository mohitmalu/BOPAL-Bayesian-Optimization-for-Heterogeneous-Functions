# BO-PAL: Boundary-Aware Partition Learning for Bayesian Optimization

Bayesian Optimization (BO) has become a standard framework for optimizing **expensive black-box functions**, where the objective is typically modeled using a stationary surrogate (e.g., Gaussian Processes). 
However, many real-world systems—such as **cyber-physical systems, machine learning pipelines, weather models, and semiconductor process optimization**—exhibit **heterogeneous and non-stationary behavior** across the input domain.

Standard BO methods struggle in such settings due to their reliance on global stationarity assumptions.

---

## 🚀 Overview

**BO-PAL (Bayesian Optimization with Partition-Aware Learning)** is a novel **gray-box Bayesian optimization framework** designed to handle **heterogeneous black-box functions**.

The key idea is to:

* Partition the input space into **locally stationary regions**
* Learn the partition structure using **boundary-aware information**
* Apply **region-specific surrogate models**
* Perform optimization using a **partition-aware acquisition strategy**

---

## 🧠 Key Contributions

* ✅ **Gray-box BO formulation** leveraging structural signals beyond function values
* ✅ **Boundary-aware partition learning** using distance-to-boundary feedback
* ✅ **MCMC-based tree sampling** for accounting the uncertainty in learned partition and adaptive partition discovery
* ✅ **PAR-GP modeling** for capturing local stationarity within regions
* ✅ **Partition-aware acquisition optimization** for efficient exploration and exploitation
* ✅ **Ensemble of Trees** for robustness against incorrectly learned partitions 
* ✅ Strong empirical performance on **synthetic and real-world benchmarks**

---

## 🔍 Problem Setting

We consider optimization of functions that:

* Are **globally non-stationary**
* Can be decomposed into **locally stationary sub-functions**
* Provide **auxiliary structural information** during evaluation:

  * Distance to the nearest region boundary

This additional information enables BO-PAL to **infer the latent partition structure**, which is otherwise unobservable in standard BO settings.

---

## ⚙️ Methodology

BO-PAL consists of three key components in each iteration:

### 1. 🌳 Partition Learning via MCMC

* Models the input space using a **tree-based partition structure**
* Uses **MCMC sampling** to grow the previously learned tree to infer region boundaries
* Incorporates **boundary distance information** for improved accuracy

### 2. 📈 Local Modeling with PAR-GP

* Applies **Partition-Aware Gaussian Processes (PAR-GP)**
* Each region is modeled with a **stationary GP**
* Captures local behavior while maintaining global flexibility

### 3. 🎯 Partition-Aware Acquisition Optimization

* Acquisition functions are evaluated **region-wise**
* Enables **targeted exploration** in uncertain regions
* Improves sample efficiency in heterogeneous spaces

---

## ▶️ Usage

Run BO-PAL optimization:

```bash
python cbo_fin_main.py
```

Key configurable parameters include:

* Number of BO iterations
* Partition depth / tree structure
* MCMC sampling parameters
* Choice of acquisition function
* Kernel specifications for GP models

Refer to `cbo_fin_main.py` and configuration files for details.

---

## 📊 Applications

BO-PAL is well-suited for:

* Semiconductor process optimization
* Hyperparameter tuning in non-stationary ML systems
* Robotics and control systems
* Scientific experiments with regime changes
* Engineering design optimization
* Falsification of Cyber-physical systems

---

## 📈 Results

BO-PAL demonstrates:

* Improved optimization performance in **heterogeneous domains**
* Better **sample efficiency** compared to standard BO methods
* Robustness to **non-stationarity and regime shifts**

---

## 📚 Citation

If you find this work useful, please cite:

```
@article{mmalu_bopal,
  title={BO-PAL: Boundary-Aware Partition Learning for Bayesian Optimization},
  author={Malu, Mohit and Pedrielli, Giulia and Dasarathy, Gautam and Spanias, Andreas},
  year={2026}
}
```

---

## 🤝 Contributions

Contributions and feedback are welcome!

* Open an issue for bugs or feature requests
* Submit a pull request for improvements

---

## 📬 Contact

For questions or collaboration:

* GitHub Issues
* Email: [mohitmalu21@gmail.com](mailto:mohitmalu21@gmail.com)

---

## ⭐ Acknowledgements

This work is motivated by the need to efficiently optimize **complex, heterogeneous systems** where traditional assumptions of stationarity break down, and where structural signals can be leveraged to accelerate learning.
